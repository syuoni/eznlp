python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr 0.01   --batch_size 32 --use_elmo False
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr 0.001  --batch_size 32 --use_elmo False
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr 0.01   --batch_size 8  --use_elmo False
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr 0.001  --batch_size 8  --use_elmo False
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr 0.01   --batch_size 32 --use_elmo True
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr 0.001  --batch_size 32 --use_elmo True
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr 0.01   --batch_size 8  --use_elmo True
python "scripts/sequence_tagging_en.py" --dataset $1 --device $2 --num_epochs 100 --lr 0.001  --batch_size 8  --use_elmo True
