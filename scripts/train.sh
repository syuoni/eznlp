python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.01
python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.001
python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.01   --batch_size 16
python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.001  --batch_size 16
python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.01   --batch_size 8
python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.001  --batch_size 8
python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.01   --use_bigram True
python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.001  --use_bigram True
python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.01   --batch_size 16 --use_bigram True
python $1 --dataset $2 --device $3 --num_epochs 100 --lr 0.001  --batch_size 16 --use_bigram True
