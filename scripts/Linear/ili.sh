# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=Time_Unet
for pred_len in 48
do
seq_len=36
while [ $seq_len -le 124 ]
do
python -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/illness \
--data_path national_illness.csv \
--model_id national_illness_$seq_len'_'$pred_len \
--model $model_name \
--data custom \
--features M \
--seq_len $seq_len \
--label_len 18 \
--pred_len $pred_len \
--enc_in 7 \
--des 'Exp' \
--stage_num 4 \
--stage_pool_kernel 3 \
--stage_pool_padding 0 \
--itr 1 --batch_size 48 --learning_rate 0.01 
let seq_len+=12
done
done

