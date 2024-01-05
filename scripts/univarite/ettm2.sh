model_name=Time_Unet

for pred_len in 336 720 
do
seq_len=144
while [ $seq_len -le 760 ]
do
  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features S \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 --batch_size 512 --learning_rate 0.005
  let seq_len+=24
done
done
