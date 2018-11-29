# Bi-LSTM+CRF Sequence Labeling

This project applied bi-lstm+crf model into named entity recognition task, which belongs to sequence labeling.

## Requirements

- Python3.5+
- PyTorch 0.4+
- numpy
- tensorboardX

## Features

- utilize `torch.utils.data.Dataset`, `torch.utils.data.Dataloader` to load corpus

- conditional random field module inside

- evaluate f1 on entity

- support both cpu & gpu

- user-friendly tensorboard visualization

- abundant arguments for tuning parameters

- nice code style & elaborate comments

## Usage

### Dataset format

for each line

```
token1 token2 token3 token4, tag1 tag2 tag3 tag4
```

please refer to `test.csv` for more detail.

### Run on GPU X

```
export CUDA_VISIBLE_DEVICES=X
python main.py --model-name MODEL_NAME --vocab token2idx.json tag2idx.json
```

### Run on CPU

```
python main.py --model-name MODEL_NAME --vocab token2idx.json tag2idx.json --no-cuda
```

## Benchmark

|config|f1|precision|recall|
|:---:|:---:|:---:|:---:|
|baseline (default config)|0.8974|0.9217|0.8743|
|lr=0.0005|0.8667|0.9|0.8357|
|embedding-size=256,hidden-size=512|0.8871|0.8929|0.8814|
|embedding-size=128,hidden-size=256|0.8895|0.9115|0.8686|
|dropout=0.1|0.9134|0.9396|0.8886|
|dropout=0.3|0.8975|0.9419|0.8571|
|dropout=0.5|0.9120|0.9367|0.8886|
|rnn-layer=2|0.9131|0.9342|0.8929|
|with-layer-norm=True|0.8943|0.9265|0.8643|


## TODO

- support customized dataset
- support multi-gpu training
- refine viterbi decoding efficiency
- extract evaluate mode