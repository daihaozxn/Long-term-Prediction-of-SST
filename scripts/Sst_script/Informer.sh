export CUDA_VISIBLE_DEVICES=0

#for d_model in 64 128 256 512
#do
#  for d_ff in 256 512 1024
#  do
#      if [ $d_model -lt $d_ff ]; then
#        python -u run.py \
#          --is_training 1 \
#          --root_path ./dataset/sst/ \
#          --data_path sst_nh.csv \
#          --model_id sst_360_360 \
#          --model Informer \
#          --data custom \
#          --features S \
#          --freq d \
#          --des 'Exp' \
#          --embed 'timeF' \
#          --enc_in 1 \
#          --dec_in 1 \
#          --c_out 1 \
#          --batch_size 32 \
#          --seq_len 360 \
#          --label_len 360 \
#          --pred_len 360 \
#          --e_layers 6 \
#          --d_layers 5 \
#          --factor 1 \
#          --train_epochs 10 \
#          --itr 3 \
#          --patience 3 \
#          --activation 'gelu' \
#          --n_heads 8 \
#          --moving_avg 45 \
#          --d_model $d_model \
#          --d_ff $d_ff
#      fi
#  done
#done
#
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/sst/ \
#  --data_path sst_nh.csv \
#  --model_id sst_360_360 \
#  --model Informer \
#  --data custom \
#  --features S \
#  --freq d \
#  --des 'Exp' \
#  --embed 'timeF' \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --batch_size 32 \
#  --seq_len 360 \
#  --label_len 360 \
#  --pred_len 360 \
#  --e_layers 6 \
#  --d_layers 5 \
#  --factor 1 \
#  --train_epochs 10 \
#  --itr 3 \
#  --patience 3 \
#  --activation 'gelu' \
#  --n_heads 8 \
#  --moving_avg 45 \
#  --d_model 512 \
#  --d_ff 2048
#
#for d_model in 64 128 256 512
#do
#  for d_ff in 256 512 1024
#  do
#      if [ $d_model -lt $d_ff ]; then
#        python -u run.py \
#          --is_training 1 \
#          --root_path ./dataset/sst/ \
#          --data_path sst_nh.csv \
#          --model_id sst_360_360 \
#          --model Informer \
#          --data custom \
#          --features M \
#          --freq d \
#          --des 'Exp' \
#          --embed 'timeF' \
#          --enc_in 25 \
#          --dec_in 25 \
#          --c_out 25 \
#          --batch_size 32 \
#          --seq_len 360 \
#          --label_len 360 \
#          --pred_len 360 \
#          --e_layers 6 \
#          --d_layers 5 \
#          --factor 1 \
#          --train_epochs 10 \
#          --itr 3 \
#          --patience 3 \
#          --activation 'gelu' \
#          --n_heads 8 \
#          --moving_avg 45 \
#          --d_model $d_model \
#          --d_ff $d_ff
#      fi
#  done
#done

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/sst/ \
  --data_path sst_nh.csv \
  --model_id sst_360_360 \
  --model Informer \
  --data custom \
  --features M \
  --freq d \
  --des 'Exp' \
  --embed 'timeF' \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --batch_size 32 \
  --seq_len 360 \
  --label_len 360 \
  --pred_len 360 \
  --e_layers 6 \
  --d_layers 5 \
  --factor 1 \
  --train_epochs 10 \
  --itr 3 \
  --patience 3 \
  --activation 'gelu' \
  --n_heads 8 \
  --moving_avg 45 \
  --d_model 512 \
  --d_ff 2048