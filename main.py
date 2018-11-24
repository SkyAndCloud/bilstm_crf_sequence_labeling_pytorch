# /usr/bin/env python
# coding=utf-8
from __future__ import unicode_literals
import argparse
import os
import json
import sys
import linecache
import pdb

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn import init
import numpy as np
# from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch BiLSTM+CRF Sequence Labeling')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--embedding-size', type=int, default=64, metavar='N',
                    help='embedding size(default: 64)')
parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                    help='hidden size(default: 256)')
parser.add_argument('--rnn-layer', type=int, default=1, metavar='N',
                    help='RNN layer num')
parser.add_argument('--dropout', type=float, default=0, metavar='RATE',
                    help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before checkpointing')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
parser.add_argument('--vocab', nargs='+', required=True, metavar='SRC_VOCAB TGT_VOCAB',
                    help='src vocab and tgt vocab')
parser.add_argument('--trainset', type=str, default=os.path.join('data', 'train.csv'), metavar='TRAINSET',
                    help='trainset path')
parser.add_argument('--testset', type=str, default=os.path.join('data', 'test.csv'), metavar='TESTSET',
                    help='testset path')

START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"
PAD = "<PAD>"
token2idx = {
  PAD: 0
}
tag2idx = {
  START_TAG: 0,
  END_TAG: 1
}

def log_sum_exp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    Compute logsumexp in a numerically stable way.
    This is mathematically equivalent to ``tensor.exp().sum(dim, keep=keepdim).log()``.
    This function is typically used for summing log probabilities.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

class CRFLayer(nn.Module):
  def __init__(self, tag_size):
    super(CRFLayer, self).__init__()
    # transition[i][j] means transition probability from j to i
    self.transition = nn.Parameter(torch.randn(tag_size, tag_size))
    # initialize START_TAG, END_TAG probability in log space
    self.transition[tag2idx[START_TAG], :] = -10000
    self.transition[:, tag2idx[END_TAG]] = -10000
    
    self.reset_parameters()

  def reset_parameters(self):
    init.normal_(self.transition.data)
  
  def forward(self, feats, mask):
    """
    Arg:
      feats: (seq_len, batch_size, tag_size)
      mask: (seq_len, batch_size)
    Return:
      scores: (batch_size, )
    """
    seq_len, batch_size, tag_size = feats.size()
    # initialize alpha to zero in log space
    alpha = torch.full((batch_size, tag_size), -10000)
    # alpha in START_TAG is 1
    alpha[:, tag2idx[START_TAG]] = 0

    for i, feat in enumerate(feats):
      # broadcast dimension: (batch_size, next_tag, current_tag)
      # emit_score is the same regardless of current_tag, so we broadcast along current_tag
      emit_score = feat.unsqueeze(-1) # (batch_size, tag_size, 1)
      # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
      transition_score = self.transition.unsqueeze(0) # (1, tag_size, tag_size)
      # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
      alpha_score = alpha.unsqueeze(1) # (batch_size, 1, tag_size)
      alpha_score = alpha_score + transition_score + emit_score
      # log_sum_exp along current_tag dimension to get next_tag alpha
      alpha = log_sum_exp(alpha_score, -1) * mask[i] + alpha * (1 - mask[i]) # (batch_size, tag_size)
    # arrive at END_TAG
    alpha = alpha + self.transition[tag2idx[END_TAG]].unsqueeze(0)

    return log_sum_exp(alpha, -1) # (batch_size, )

  def score_sentence(self, feats, tags, mask):
    """
    Arg:
      feats: (seq_len, batch_size, tag_size)
      tags: (seq_len, batch_size)
      mask: (seq_len, batch_size)
    Return:
      scores: (batch_size, )
    """
    seq_len, batch_size, tag_size = feats.size()
    scores = torch.zeros(batch_size)
    tags = torch.cat([torch.full((1, batch_size), tag2idx[START_TAG], dtype=torch.long), tags], 0) # (seq_len + 1, batch_size)
    for i, feat in enumerate(feats):
      emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[i + 1])])
      transition_score = torch.stack([self.transition[tags[i + 1, b], tags[i, b]] for b in range(batch_size)])
      scores += (emit_score + transition_score) * mask[i]
    transition_to_end = torch.stack([self.transition[tag2idx[END_TAG], tag[mask[:, b].sum().long()]] for b, tag in enumerate(tags.transpose(0, 1))])
    scores += transition_to_end
    return scores

  def viterbi_decode(self, feats, mask):
    """
    :param feats: (seq_len, batch_size, tag_size)
    :param mask: (seq_len, batch_size)
    :return best_path: (seq_len, batch_size)
    """
    seq_len, batch_size, tag_size = feats.size()
    # initialize scores in log space
    scores = torch.fill((batch_size, tag_size), -10000)
    scores[:, tag2idx[START_TAG]] = 0
    pointers = []
    # forward
    for i, feat in enumerate(feats):
      # broadcast dimension: (batch_size, next_tag, current_tag)
      scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)  # (batch_size, tag_size, tag_size)
      # max along current_tag to obtain: next_tag score, current_tag pointer
      scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, tag_size), (batch_size, tag_size)
      scores_t += feat
      pointers.append(pointer)
      mask_t = mask[i].unsqueeze(-1)  # (batch_size, 1)
      scores = scores_t * mask_t + scores * mask_t
    # pointers should be list with shape (seq_len, batch_size, tag_size)
    scores += self.transition[tag2idx[END_TAG]].unsqueeze(0)
    best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )
    # backtracking
    best_path = best_tag.unsqueeze(-1).tolist() # list shape (batch_size, 1)
    for i in range(batch_size):
      best_tag_i = best_tag[i]
      seq_len_i = mask[:, i].sum()
      for ptr_t in reversed(pointers[:seq_len_i][i]):
        # ptr_t shape (tag_size, )
        best_tag_i = ptr_t[best_tag_i]
        best_path[i].append(best_tag_i)
      # pop first tag
      best_path[i].pop()
      # reverse order
      best_path[i].reverse()
    return best_path

class BiLSTMCRF(nn.Module):
  def __init__(self, vocab_size, tag_size, embedding_size, hidden_size, num_layers, dropout):
    super(BiLSTMCRF, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=token2idx[PAD])
    self.bilstm = nn.LSTM(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=True)
    self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)
    self.crf = CRFLayer(tag_size)
    
    self.reset_parameters()

  def reset_parameters(self):
    self.crf.reset_parameters()
    self.bilstm.reset_parameters()
    init.xavier_normal_(self.embedding.weight)
    init.xavier_normal_(self.hidden2tag.weight)

  def get_lstm_features(self, seq, mask):
    embed = self.embedding(seq) # (seq_len, batch_size, embedding_size)
    lstm_output, _ = self.bilstm(embed) # (seq_len, batch_size, hidden_size)
    lstm_features = self.hidden2tag(lstm_output) * mask[:, :, None]  # (seq_len, batch_size, tag_size)
    return lstm_features

  def neg_log_likelihood(self, seq, tags, mask):
    lstm_features = self.get_lstm_features(seq, mask)
    forward_score = self.crf(lstm_features, mask)
    gold_score = self.crf.score_sentence(lstm_features, tags, mask)
    return forward_score - gold_score

  def predict(self, seq):
    lstm_features = self.get_lstm_features(seq)
    score, prediction = self.crf.viterbi_decode(lstm_features)
    return score, prediction

class SequenceLabelingDataset(Dataset):
  def __init__(self, filename):
    self._filename = filename
    with open(filename, "r", encoding="utf-8") as f:
      self._lines_count = len(f.readlines())

  def __getitem__(self, idx):
    line = linecache.getline(self._filename, idx)
    return line.strip().split(",")

  def __len__(self):
    return self._lines_count

def main(args):
  global START_TAG
  global END_TAG
  global PAD
  global token2idx
  global tag2idx

  use_cuda = torch.cuda.is_available() and not args.no_cuda
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.manual_seed(args.seed)
  if use_cuda:
    torch.cuda.manual_seed(args.seed)

  if len(args.vocab) != 2:
    print("ERROR: invalid vocab arguments -> {}".format(args.vocab), file=sys.stderr)
    exit(-1)

  if os.path.exists(args.vocab[0]) and os.path.exists(args.vocab[1]):
    with open(args.vocab[0], "r", encoding="utf-8") as fp:
      token2idx = json.load(fp)
    with open(args.vocab[1], "r", encoding="utf-8") as fp:
      tag2idx = json.load(fp)
  else:
    with open(args.trainset, "r", encoding="utf-8") as fp:
      for cursor in fp.readlines():
        seq, tags = cursor.split(',')
        for tok in seq.split():
          if tok not in token2idx:
            token2idx[tok] = len(token2idx)
        for tag in tags.split():
          if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)
    with open(args.vocab[0], "w", encoding="utf-8") as fp:
      json.dump(token2idx, fp, ensure_ascii=False)
    with open(args.vocab[1], "w", encoding="utf-8") as fp:
      json.dump(tag2idx, fp)

  trainset = SequenceLabelingDataset(args.trainset)
  testset = SequenceLabelingDataset(args.testset)
  trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
  testset_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

  model = BiLSTMCRF(len(token2idx), len(tag2idx), args.embedding_size, args.hidden_size, args.rnn_layer, args.dropout).to(device)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  model.train()
  step = 0

  def _prepare_data(samples, vocab, pad, device=None):
    samples = list(map(lambda s: s.replace(" ", ""), samples))
    batch_size = len(samples)
    sizes = [len(s) for s in samples]
    max_size = max(sizes)
    x_np = np.full((batch_size, max_size), fill_value=vocab[pad], dtype='int64')
    for i in range(batch_size):
      x_np[i, :sizes[i]] = [vocab[token] for token in samples[i]]
    return torch.LongTensor(x_np.T).to(device)

  for eidx in range(1, args.epochs + 1):
    for bidx, batch in enumerate(trainset_loader):
      seq = _prepare_data(batch[0], token2idx, PAD, device)
      tags = _prepare_data(batch[1], tag2idx, END_TAG, device)
      mask = torch.ne(seq, float(token2idx[PAD])).float()
      optimizer.zero_grad()
      loss = model.neg_log_likelihood(seq, tags, mask)
      loss.backward()
      optimizer.step()
      step += 1

if __name__ == "__main__":
  main(parser.parse_args())