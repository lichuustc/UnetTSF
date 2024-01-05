# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336

model_name=Time_Unet
python -u run_longExp.py \
          --is_training 1 \
          --root_path ./dataset/ETT \
          --data_path ETTm1.csv \
          --model_id ETTm1_$seq_len'_'336 \
          --model $model_name \
          --data ETTm1 \
          --features M \
          --seq_len $seq_len \
          --pred_len 336 \
          --enc_in 7 \
          --des 'Exp' \
          --stage_num 4 \
          --stage_pool_kernel 3 \
          --stage_pool_padding 0 \
          --itr 1 --batch_size 256 --learning_rate 0.01 

python -u run_longExp.py \
          --is_training 1 \
          --root_path ./dataset/ETT \
          --data_path ETTm1.csv \
          --model_id ETTm1_$seq_len'_'720 \
          --model $model_name \
          --data ETTm1 \
          --features M \
          --seq_len $seq_len \
          --pred_len 720 \
          --enc_in 7 \
          --des 'Exp' \
          --stage_num 4 \
          --stage_pool_kernel 3 \
          --stage_pool_padding 0 \
          --itr 1 --batch_size 256 --learning_rate 0.01 


