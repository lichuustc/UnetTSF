model_name=Time_Unet

for pred_len in 96 192 336 720
do
seq_len=96

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --stage_num 5 \
  --stage_pool_kernel 3 \
  --stage_pool_padding 0 \
  --itr 1 --batch_size 32 --learning_rate 0.05
done