# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=Time_Unet

for stage_pool_kernel in 3 5 7
do
    for stage_num in 2 3 4
    do
        python -u run_longExp.py \
          --is_training 1 \
          --root_path ./dataset/ETT \
          --data_path ETTm2.csv \
          --model_id ETTm2_$seq_len'_'96 \
          --model $model_name \
          --data ETTm2 \
          --features M \
          --seq_len $seq_len \
          --pred_len 96 \
          --enc_in 7 \
          --des 'Exp' \
          --stage_num $stage_num \
          --stage_pool_kernel $stage_pool_kernel \
          --stage_pool_padding 0 \
          --itr 1 --batch_size 256 --learning_rate 0.01 
    done
done


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.001 
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.1 
