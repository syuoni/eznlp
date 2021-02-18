#!/bin/bash

python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --bert_lr $4 --batch_size 64 --use_bert
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --bert_lr $4 --batch_size 16 --use_bert
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --bert_lr $4 --batch_size 64 --use_bert --dec_arch SoftMax
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --bert_lr $4 --batch_size 16 --use_bert --dec_arch SoftMax
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --bert_lr $4 --batch_size 64 --use_bert --use_bert_intermediate
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --bert_lr $4 --batch_size 16 --use_bert --use_bert_intermediate
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --bert_lr $4 --batch_size 64 --use_bert --use_bert_intermediate --dec_arch SoftMax
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --bert_lr $4 --batch_size 16 --use_bert --use_bert_intermediate --dec_arch SoftMax
