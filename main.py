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
from torch.utils.data import DataLoader, Dataset
from torch.nn import init
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch BiLSTM+CRF Sequence Labeling')
parser.add_argument('--model-name', type=str, default='model', metavar='S',
                    help='model name')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='test batch size')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--embedding-size', type=int, default=512, metavar='N',
                    help='embedding size')
parser.add_argument('--hidden-size', type=int, default=1024, metavar='N',
                    help='hidden size')
parser.add_argument('--rnn-layer', type=int, default=1, metavar='N',
                    help='RNN layer num')
parser.add_argument('--with-layer-norm', action='store_true', default=False,
                    help='whether to add layer norm after RNN')
parser.add_argument('--dropout', type=float, default=0, metavar='RATE',
                    help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed')
parser.add_argument('--save-interval', type=int, default=30, metavar='N',
                    help='save interval')
parser.add_argument('--valid-interval', type=int, default=60, metavar='N',
                    help='valid interval')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='log interval')
parser.add_argument('--patience', type=int, default=30, metavar='N',
                    help='patience for early stop')
parser.add_argument('--vocab', nargs='+', required=True, metavar='SRC_VOCAB TGT_VOCAB',
                    help='src vocab and tgt vocab')
parser.add_argument('--trainset', type=str, default=os.path.join('data', 'train.csv'), metavar='TRAINSET',
                    help='trainset path')
parser.add_argument('--testset', type=str, default=os.path.join('data', 'test.csv'), metavar='TESTSET',
                    help='testset path')

START_TAG = "<START_TAG>"
END_TAG = "<END_TAG>"
O = "O"
BLOC = "B-LOC"
ILOC = "I-LOC"
BORG = "B-ORG"
IORG = "I-ORG"
BPER = "B-PER"
IPER = "I-PER"
PAD = "<PAD>"
UNK = "<UNK>"
token2idx = {
  PAD: 0,
  UNK: 1
}
tag2idx = {
  START_TAG: 0,
  END_TAG: 1,
  O: 2,
  BLOC: 3,
  ILOC: 4,
  BORG: 5,
  IORG: 6,
  BPER: 7,
  IPER: 8
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

    self.reset_parameters()

  def reset_parameters(self):
    init.normal_(self.transition)
    # initialize START_TAG, END_TAG probability in log space
    self.transition.detach()[tag2idx[START_TAG], :] = -10000
    self.transition.detach()[:, tag2idx[END_TAG]] = -10000

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
    alpha = feats.new_full((batch_size, tag_size), fill_value=-10000)
    # alpha in START_TAG is 1
    alpha[:, tag2idx[START_TAG]] = 0
    for t, feat in enumerate(feats):
      # broadcast dimension: (batch_size, next_tag, current_tag)
      # emit_score is the same regardless of current_tag, so we broadcast along current_tag
      emit_score = feat.unsqueeze(-1) # (batch_size, tag_size, 1)
      # transition_score is the same regardless of each sample, so we broadcast along batch_size dimension
      transition_score = self.transition.unsqueeze(0) # (1, tag_size, tag_size)
      # alpha_score is the same regardless of next_tag, so we broadcast along next_tag dimension
      alpha_score = alpha.unsqueeze(1) # (batch_size, 1, tag_size)
      alpha_score = alpha_score + transition_score + emit_score
      # log_sum_exp along current_tag dimension to get next_tag alpha
      mask_t = mask[t].unsqueeze(-1)
      alpha = log_sum_exp(alpha_score, -1) * mask_t + alpha * (1 - mask_t) # (batch_size, tag_size)
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
    scores = feats.new_zeros(batch_size)
    tags = torch.cat([tags.new_full((1, batch_size), fill_value=tag2idx[START_TAG]), tags], 0) # (seq_len + 1, batch_size)
    for t, feat in enumerate(feats):
      emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[t + 1])])
      transition_score = torch.stack([self.transition[tags[t + 1, b], tags[t, b]] for b in range(batch_size)])
      scores += (emit_score + transition_score) * mask[t]
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
    scores = feats.new_full((batch_size, tag_size), fill_value=-10000)
    scores[:, tag2idx[START_TAG]] = 0
    pointers = []
    # forward
    for t, feat in enumerate(feats):
      # broadcast dimension: (batch_size, next_tag, current_tag)
      scores_t = scores.unsqueeze(1) + self.transition.unsqueeze(0)  # (batch_size, tag_size, tag_size)
      # max along current_tag to obtain: next_tag score, current_tag pointer
      scores_t, pointer = torch.max(scores_t, -1)  # (batch_size, tag_size), (batch_size, tag_size)
      scores_t += feat
      pointers.append(pointer)
      mask_t = mask[t].unsqueeze(-1)  # (batch_size, 1)
      scores = scores_t * mask_t + scores * (1 - mask_t)
    pointers = torch.stack(pointers, 0) # (seq_len, batch_size, tag_size)
    scores += self.transition[tag2idx[END_TAG]].unsqueeze(0)
    best_score, best_tag = torch.max(scores, -1)  # (batch_size, ), (batch_size, )
    # backtracking
    best_path = best_tag.unsqueeze(-1).tolist() # list shape (batch_size, 1)
    for i in range(batch_size):
      best_tag_i = best_tag[i]
      seq_len_i = int(mask[:, i].sum())
      for ptr_t in reversed(pointers[:seq_len_i, i]):
        # ptr_t shape (tag_size, )
        best_tag_i = ptr_t[best_tag_i].item()
        best_path[i].append(best_tag_i)
      # pop first tag
      best_path[i].pop()
      # reverse order
      best_path[i].reverse()
    return best_path

class BiLSTMCRF(nn.Module):
  def __init__(self, vocab_size, tag_size, embedding_size, hidden_size, num_layers, dropout, with_ln):
    super(BiLSTMCRF, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=token2idx[PAD])
    self.dropout = nn.Dropout(dropout)
    self.bilstm = nn.LSTM(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=True)
    self.with_ln = with_ln
    if with_ln:
      self.layer_norm = nn.LayerNorm(hidden_size)
    self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)
    self.crf = CRFLayer(tag_size)

    self.reset_parameters()

  def reset_parameters(self):
    init.xavier_normal_(self.embedding.weight)
    init.xavier_normal_(self.hidden2tag.weight)

  def get_lstm_features(self, seq, mask):
    """
    :param seq: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    """
    embed = self.embedding(seq) # (seq_len, batch_size, embedding_size)
    embed = self.dropout(embed)
    embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(0).long())
    lstm_output, _ = self.bilstm(embed) # (seq_len, batch_size, hidden_size)
    lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output)
    lstm_output = lstm_output * mask.unsqueeze(-1)
    if self.with_ln:
      lstm_output = self.layer_norm(lstm_output)
    lstm_features = self.hidden2tag(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, tag_size)
    return lstm_features

  def neg_log_likelihood(self, seq, tags, mask):
    """
    :param seq: (seq_len, batch_size)
    :param tags: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    """
    lstm_features = self.get_lstm_features(seq, mask)
    forward_score = self.crf(lstm_features, mask)
    gold_score = self.crf.score_sentence(lstm_features, tags, mask)
    loss = (forward_score - gold_score).sum()
    #loss = -self.crf(lstm_features.transpose(0, 1), tags.transpose(0, 1), mask.transpose(0, 1))

    # for bilstm model
    # log_probs = nn.functional.log_softmax(lstm_features, dim=-1).view(-1, self.tag_size)
    # tags = tags.view(-1)
    # loss = nn.functional.nll_loss(log_probs, tags, ignore_index=tag2idx[END_TAG], reduction="sum")

    return loss

  def predict(self, seq, mask):
    """
    :param seq: (seq_len, batch_size)
    :param mask: (seq_len, batch_size)
    """
    lstm_features = self.get_lstm_features(seq, mask)
    best_paths = self.crf.viterbi_decode(lstm_features, mask)
    #return best_path
    #best_paths = self.crf.viterbi_tags(lstm_features.transpose(0, 1), mask.transpose(0, 1).long())

    # for bilstm model
    # best_paths = torch.max(lstm_features, -1)[1] * mask.long()
    # best_paths = best_paths.transpose(0, 1).tolist()

    return best_paths

class SequenceLabelingDataset(Dataset):
  def __init__(self, filename):
    self._filename = filename
    with open(filename, "r", encoding="utf-8") as f:
      self._lines_count = len(f.readlines())

  def __getitem__(self, idx):
    line = linecache.getline(self._filename, idx + 1)
    return line.strip().split(",")

  def __len__(self):
    return self._lines_count

def main(args):
  global START_TAG
  global END_TAG
  global PAD
  global token2idx
  global tag2idx

  tb_writer = SummaryWriter(args.model_name)

  print("Args: {}".format(args))
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
  idx2tag = {}
  for k, v in tag2idx.items():
    idx2tag[v] = k
  idx2token = {}
  for k, v in token2idx.items():
    idx2token[v] = k

  print("Loading data")
  trainset = SequenceLabelingDataset(args.trainset)
  testset = SequenceLabelingDataset(args.testset)
  trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
  testset_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)

  print("Building model")
  model = BiLSTMCRF(len(token2idx), len(tag2idx), args.embedding_size, args.hidden_size, args.rnn_layer, args.dropout,
          args.with_layer_norm).to(device)
  print(model)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  print("Start training")
  model.train()
  step = 0

  def _prepare_data(samples, vocab, pad, device=None):
    samples = list(map(lambda s: s.strip().split(" "), samples))
    batch_size = len(samples)
    sizes = [len(s) for s in samples]
    max_size = max(sizes)
    x_np = np.full((batch_size, max_size), fill_value=vocab[pad], dtype='int64')
    for i in range(batch_size):
      x_np[i, :sizes[i]] = [vocab[token] if token in vocab else vocab[UNK] for token in samples[i]]
    return torch.LongTensor(x_np.T).to(device)

  def _compute_forward(seq, tags, mask):
    loss = model.neg_log_likelihood(seq, tags, mask)
    batch_size = seq.size(1)
    loss /= batch_size
    loss.backward()
    return loss.item()

  def _evaluate():
    def get_entity(tags):
      entity = []
      prev_entity = "O"
      start = -1
      end = -1
      for i, tag in enumerate(tags):
        if tag[0] == "O":
          if prev_entity != "O":
            entity.append((start, end))
          prev_entity = "O"
        if tag[0] == "B":
          if prev_entity != "O":
            entity.append((start, end))
          prev_entity = tag[2:]
          start = end = i
        if tag[0] == "I":
          if prev_entity == tag[2:]:
            end = i
      return entity

    model.eval()
    correct_num = 0
    predict_num = 0
    truth_num = 0
    with torch.no_grad():
      for bidx, batch in enumerate(testset_loader):
        seq = _prepare_data(batch[0], token2idx, PAD, device)
        mask = torch.ne(seq, float(token2idx[PAD])).float()
        length = mask.sum(0)
        _, idx = length.sort(0, descending=True)
        seq = seq[:, idx]
        mask = mask[:, idx]
        best_path = model.predict(seq, mask)
        ground_truth = [batch[1][i].strip().split(" ") for i in idx]
        for hyp, gold in zip(best_path, ground_truth):
          hyp = list(map(lambda x: idx2tag[x], hyp))
          predict_entities = get_entity(hyp)
          gold_entities = get_entity(gold)
          correct_num += len(set(predict_entities) & set(gold_entities))
          predict_num += len(set(predict_entities))
          truth_num += len(set(gold_entities))
    # calculate F1 on entity
    precision = correct_num / predict_num if predict_num else 0
    recall = correct_num / truth_num if truth_num else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    model.train()
    return f1, precision, recall

  best_f1 = 0
  patience = 0
  early_stop = False
  for eidx in range(1, args.epochs + 1):
    if eidx == 2:
      model.debug = True
    if early_stop:
      print("Early stop. epoch {} step {} best f1 {}".format(eidx, step, best_f1))
      sys.exit(0)
    print("Start epoch {}".format(eidx))
    for bidx, batch in enumerate(trainset_loader):
      seq = _prepare_data(batch[0], token2idx, PAD, device)
      tags = _prepare_data(batch[1], tag2idx, END_TAG, device)
      mask = torch.ne(seq, float(token2idx[PAD])).float()
      length = mask.sum(0)
      _, idx = length.sort(0, descending=True)
      seq = seq[:, idx]
      tags = tags[:, idx]
      mask = mask[:, idx]
      optimizer.zero_grad()
      loss = _compute_forward(seq, tags, mask)
      tb_writer.add_scalar("train/loss", loss, step)
      tb_writer.add_scalar("train/epoch", step, eidx)
      optimizer.step()
      step += 1
      if step % args.log_interval == 0:
        print("epoch {} step {} batch {} loss {}".format(eidx, step, bidx, loss))
      if step % args.save_interval == 0:
        torch.save(model.state_dict(), os.path.join(args.model_name, "newest.model"))
        torch.save(optimizer.state_dict(), os.path.join(args.model_name, "newest.optimizer"))
      if step % args.valid_interval == 0:
        f1, precision, recall = _evaluate()
        tb_writer.add_scalar("eval/f1", f1, step)
        tb_writer.add_scalar("eval/precision", precision, step)
        tb_writer.add_scalar("eval/recall", recall, step)
        print("[valid] epoch {} step {} f1 {} precision {} recall {}".format(eidx, step, f1, precision, recall))
        if f1 > best_f1:
          patience = 0
          best_f1 = f1
          torch.save(model.state_dict(), os.path.join(args.model_name, "best.model"))
          torch.save(optimizer.state_dict(), os.path.join(args.model_name, "best.optimizer"))
        else:
          patience += 1
          if patience == args.patience:
            early_stop = True

if __name__ == "__main__":
  main(parser.parse_args())
