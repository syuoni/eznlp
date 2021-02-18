#!/bin/bash

python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 64
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 16
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 64 --use_elmo
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 16 --use_elmo
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 64 --use_flair
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 16 --use_flair
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 64 --optimizer SGD --scheduler ReduceLROnPlateau 
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 16 --optimizer SGD --scheduler ReduceLROnPlateau
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 64 --optimizer SGD --scheduler ReduceLROnPlateau --use_elmo
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 16 --optimizer SGD --scheduler ReduceLROnPlateau --use_elmo
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 64 --optimizer SGD --scheduler ReduceLROnPlateau --use_flair
python scripts/sequence_tagging_en.py --dataset $1 --device $2 --lr $3 --batch_size 16 --optimizer SGD --scheduler ReduceLROnPlateau --use_flair
