python "scripts/sequence_tagging_zh.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64
python "scripts/sequence_tagging_zh.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16
python "scripts/sequence_tagging_zh.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64 --use_bigram
python "scripts/sequence_tagging_zh.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16 --use_bigram
python "scripts/sequence_tagging_zh.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64 --emb_freeze
python "scripts/sequence_tagging_zh.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16 --emb_freeze
python "scripts/sequence_tagging_zh.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 64 --use_bigram --emb_freeze
python "scripts/sequence_tagging_zh.py" --dataset $1 --device $2 --num_epochs 100 --lr $3 --batch_size 16 --use_bigram --emb_freeze