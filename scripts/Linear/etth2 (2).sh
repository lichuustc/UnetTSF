model_name=Time_Unet

for pred_len in 96 192 336 720
do
for seq_len in 48 72 96 120 144 168 192 336 504 672 720
do
  
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
  --itr 1 --batch_size 32 --learning_rate 0.05

done
done

for pred_len in in 96 192 336 720
do
for seq_len in 36 48 60 72 144 288
do

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LookBackWindow/$model_name'_'ETTm2_$seq_len'_'$pred_len.log
done
done

