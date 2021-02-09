#!/bin/bash

python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64 --use_elmo
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16 --use_elmo
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64 --use_flair
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16 --use_flair
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64 --emb_freeze
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16 --emb_freeze
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64 --use_elmo --emb_freeze
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16 --use_elmo --emb_freeze
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64 --use_flair --emb_freeze
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16 --use_flair --emb_freeze
