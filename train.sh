#！/bin/bash
mkdir default && python main.py --model-name default --vocab token2idx.json tag2idx.json 2>&1 | tee default/log
mkdir train1 && python main.py --model-name train1 --vocab token2idx.json tag2idx.json --lr 0.0005 2>&1 | tee train1/log
mkdir train2 && python main.py --model-name train2 --vocab token2idx.json tag2idx.json --embedding-size 256 --hidden-size 512 2>&1 | tee train2/log
mkdir train3 && python main.py --model-name train3 --vocab token2idx.json tag2idx.json --embedding-size 128 --hidden-size 256 2>&1 | tee train3/log
mkdir train4 && python main.py --model-name train4 --vocab token2idx.json tag2idx.json --dropout 0.1 2>&1 | tee train4/log
mkdir train5 && python main.py --model-name train5 --vocab token2idx.json tag2idx.json --dropout 0.3 2>&1 | tee train5/log
mkdir train6 && python main.py --model-name train6 --vocab token2idx.json tag2idx.json --dropout 0.5 2>&1 | tee train6/log
mkdir train7 && python main.py --model-name train7 --vocab token2idx.json tag2idx.json --rnn-layer 2 2>&1 | tee train7/log
mkdir train8 && python main.py --model-name train8 --vocab token2idx.json tag2idx.json --with-layer-norm 2 2>&1 | tee train8/log
