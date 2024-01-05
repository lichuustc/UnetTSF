model_name=Time_Unet

for pred_len in 336 720
do
seq_len=144
while [ $seq_len -le 760 ]
do
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ETT \
    --data_path ETTh1.csv \
    --model_id ETTh1_$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features S \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 1 \
    --des 'Exp' \
    --stage_num 4 \
    --stage_pool_kernel 3 \
    --stage_pool_padding 0 \
    --itr 1 --batch_size 256 --learning_rate 0.01 
  let seq_len+=24
done
done