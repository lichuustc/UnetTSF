model_name=Time_Unet

for pred_len in 720
do
for seq_len in  192 336 504 672 720
do
  
  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/weather \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 128  

done
done








