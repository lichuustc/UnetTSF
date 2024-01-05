python -u .\run_longExp.py --model PatchTST --pred_len 96 --seq_len 336 --batch_size 32    --e_layers 3  --n_heads 4  --d_model 16   --d_ff 128  --dropout 0.3 --fc_dropout 0.3  --head_dropout 0 --patch_len 16  --stride 8

python -u .\run_longExp.py --model Time_Unet --pred_len 96 --seq_len 336 --batch_size 32

python -u .\run_longExp.py --model DLinear --pred_len 96 --seq_len 336 --batch_size 32 


python -u .\run_longExp.py --model Autoformer --pred_len 96 --seq_len 336 --batch_size 32 

python -u .\run_longExp.py --model Informer --pred_len 96 --seq_len 336 --batch_size 32 