export CUDA_VISIBLE_DEVICES=0

for features in S M
do
  if [ $features == S ];then
    enc_in=1
    dec_in=1
    c_out=1
  else
    enc_in=25
    dec_in=25
    c_out=25
  fi
  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/sst/ \
  --data_path sst_dh.csv \
  --model_id sst_dh \
  --model TransSt \
  --data custom \
  --features $features \
  --freq d \
  --des 'Exp' \
  --embed 'timeF' \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --seq_len 360 \
  --label_len 330 \
  --pred_len 30 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --train_epochs 10 \
  --itr 3 \
  --patience 3 \
  --activation 'gelu' \
  --n_heads 8 \
  --moving_avg 45 \
  --d_model 256 \
  --d_ff 1024

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/sst/ \
  --data_path sst_dh.csv \
  --model_id sst_dh \
  --model TransSt \
  --data custom \
  --features $features \
  --freq d \
  --des 'Exp' \
  --embed 'timeF' \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --seq_len 360 \
  --label_len 300 \
  --pred_len 60 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --train_epochs 10 \
  --itr 3 \
  --patience 3 \
  --activation 'gelu' \
  --n_heads 8 \
  --moving_avg 45 \
  --d_model 256 \
  --d_ff 1024

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/sst/ \
  --data_path sst_dh.csv \
  --model_id sst_dh \
  --model TransSt \
  --data custom \
  --features $features \
  --freq d \
  --des 'Exp' \
  --embed 'timeF' \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --seq_len 360 \
  --label_len 270 \
  --pred_len 90 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --train_epochs 10 \
  --itr 3 \
  --patience 3 \
  --activation 'gelu' \
  --n_heads 8 \
  --moving_avg 45 \
  --d_model 256 \
  --d_ff 1024

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/sst/ \
  --data_path sst_dh.csv \
  --model_id sst_dh \
  --model TransSt \
  --data custom \
  --features $features \
  --freq d \
  --des 'Exp' \
  --embed 'timeF' \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --seq_len 360 \
  --label_len 180 \
  --pred_len 180 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --train_epochs 10 \
  --itr 3 \
  --patience 3 \
  --activation 'gelu' \
  --n_heads 8 \
  --moving_avg 45 \
  --d_model 256 \
  --d_ff 1024

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/sst/ \
  --data_path sst_dh.csv \
  --model_id sst_dh \
  --model TransSt \
  --data custom \
  --features $features \
  --freq d \
  --des 'Exp' \
  --embed 'timeF' \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --seq_len 360 \
  --label_len 360 \
  --pred_len 270 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --train_epochs 10 \
  --itr 3 \
  --patience 3 \
  --activation 'gelu' \
  --n_heads 8 \
  --moving_avg 45 \
  --d_model 256 \
  --d_ff 1024

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/sst/ \
  --data_path sst_dh.csv \
  --model_id sst_dh \
  --model TransSt \
  --data custom \
  --features $features \
  --freq d \
  --des 'Exp' \
  --embed 'timeF' \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 32 \
  --seq_len 360 \
  --label_len 360 \
  --pred_len 360 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --train_epochs 10 \
  --itr 3 \
  --patience 3 \
  --activation 'gelu' \
  --n_heads 8 \
  --moving_avg 45 \
  --d_model 256 \
  --d_ff 1024
done


#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_bh.csv \
#--model_id sst_bh \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers 3 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024


#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_bh.csv \
#--model_id sst_bh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 300 \
#--pred_len 60 \
#--e_layers 3 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_bh.csv \
#--model_id sst_bh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers 3 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 360 \
#--pred_len 360 \
#--e_layers 3 \
#--d_layers 2 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_hh.csv \
#--model_id sst_hh \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers 4 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_dh.csv \
#--model_id sst_dh \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers 4 \
#--d_layers 2 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_dh.csv \
#--model_id sst_dh \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers 3 \
#--d_layers 2 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_dh.csv \
#--model_id sst_dh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers 3 \
#--d_layers 2 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_nh.csv \
#--model_id sst_nh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers 4 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_nh.csv \
#--model_id sst_nh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers 4 \
#--d_layers 2 \
#--factor 3 \
#--train_epochs 10 \
#--itr 12 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024

#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_bh.csv \
#--model_id sst_bh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers 4 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 360 \
#--pred_len 360 \
#--e_layers 4 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers 4 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 360 \
#--pred_len 360 \
#--e_layers 3 \
#--d_layers 2 \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 540 \
#--label_len 360 \
#--pred_len 360 \
#--e_layers 4 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_bh.csv \
#--model_id sst_bh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 300 \
#--pred_len 60 \
#--e_layers 3 \
#--d_layers 2 \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_bh.csv \
#--model_id sst_bh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers 3 \
#--d_layers 2 \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_bh.csv \
#--model_id sst_bh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers 3 \
#--d_layers 2 \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_bh.csv \
#--model_id sst_bh \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers 4 \
#--d_layers 1 \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024


#for e_layers in 5 6
#do
#  for d_layers in 1 2
#  do
#    python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/sst/ \
#    --data_path sst_bh.csv \
#    --model_id sst_bh \
#    --model TransSt \
#    --data custom \
#    --features S \
#    --freq d \
#    --des 'Exp' \
#    --embed 'timeF' \
#    --enc_in 1 \
#    --dec_in 1 \
#    --c_out 1 \
#    --batch_size 32 \
#    --seq_len 360 \
#    --label_len 270 \
#    --pred_len 90 \
#    --e_layers $e_layers \
#    --d_layers $d_layers \
#    --factor 3 \
#    --train_epochs 10 \
#    --itr 6 \
#    --patience 3 \
#    --activation 'gelu' \
#    --n_heads 8 \
#    --moving_avg 45 \
#    --d_model 256 \
#    --d_ff 1024

#    python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/sst/ \
#    --data_path sst_tw.csv \
#    --model_id sst_tw \
#    --model TransSt \
#    --data custom \
#    --features S \
#    --freq d \
#    --des 'Exp' \
#    --embed 'timeF' \
#    --enc_in 1 \
#    --dec_in 1 \
#    --c_out 1 \
#    --batch_size 32 \
#    --seq_len 360 \
#    --label_len 300 \
#    --pred_len 60 \
#    --e_layers $e_layers \
#    --d_layers $d_layers \
#    --factor 3 \
#    --train_epochs 10 \
#    --itr 6 \
#    --patience 3 \
#    --activation 'gelu' \
#    --n_heads 8 \
#    --moving_avg 45 \
#    --d_model 256 \
#    --d_ff 1024
#
#    python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/sst/ \
#    --data_path sst_tw.csv \
#    --model_id sst_tw \
#    --model TransSt \
#    --data custom \
#    --features S \
#    --freq d \
#    --des 'Exp' \
#    --embed 'timeF' \
#    --enc_in 1 \
#    --dec_in 1 \
#    --c_out 1 \
#    --batch_size 32 \
#    --seq_len 360 \
#    --label_len 180 \
#    --pred_len 180 \
#    --e_layers $e_layers \
#    --d_layers $d_layers \
#    --factor 3 \
#    --train_epochs 10 \
#    --itr 6 \
#    --patience 3 \
#    --activation 'gelu' \
#    --n_heads 8 \
#    --moving_avg 45 \
#    --d_model 256 \
#    --d_ff 1024

#    done
#done

#for e_layers in 5 6
#do
#  for d_layers in 1 2
#  do
#    python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/sst/ \
#    --data_path sst_bh.csv \
#    --model_id sst_bh \
#    --model TransSt \
#    --data custom \
#    --features M \
#    --freq d \
#    --des 'Exp' \
#    --embed 'timeF' \
#    --enc_in 25 \
#    --dec_in 25 \
#    --c_out 25 \
#    --batch_size 32 \
#    --seq_len 360 \
#    --label_len 300 \
#    --pred_len 60 \
#    --e_layers $e_layers \
#    --d_layers $d_layers \
#    --factor 3 \
#    --train_epochs 10 \
#    --itr 6 \
#    --patience 3 \
#    --activation 'gelu' \
#    --n_heads 8 \
#    --moving_avg 45 \
#    --d_model 256 \
#    --d_ff 1024
#
#    python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/sst/ \
#    --data_path sst_bh.csv \
#    --model_id sst_bh \
#    --model TransSt \
#    --data custom \
#    --features M \
#    --freq d \
#    --des 'Exp' \
#    --embed 'timeF' \
#    --enc_in 25 \
#    --dec_in 25 \
#    --c_out 25 \
#    --batch_size 32 \
#    --seq_len 360 \
#    --label_len 270 \
#    --pred_len 90 \
#    --e_layers $e_layers \
#    --d_layers $d_layers \
#    --factor 3 \
#    --train_epochs 10 \
#    --itr 6 \
#    --patience 3 \
#    --activation 'gelu' \
#    --n_heads 8 \
#    --moving_avg 45 \
#    --d_model 256 \
#    --d_ff 1024

#    python -u run.py \
#    --is_training 1 \
#    --root_path ./dataset/sst/ \
#    --data_path sst_tw.csv \
#    --model_id sst_tw \
#    --model TransSt \
#    --data custom \
#    --features M \
#    --freq d \
#    --des 'Exp' \
#    --embed 'timeF' \
#    --enc_in 25 \
#    --dec_in 25 \
#    --c_out 25 \
#    --batch_size 32 \
#    --seq_len 360 \
#    --label_len 360 \
#    --pred_len 360 \
#    --e_layers $e_layers \
#    --d_layers $d_layers \
#    --factor 3 \
#    --train_epochs 10 \
#    --itr 6 \
#    --patience 3 \
#    --activation 'gelu' \
#    --n_heads 8 \
#    --moving_avg 45 \
#    --d_model 256 \
#    --d_ff 1024

#    done
#done

#e_layers=3
#d_layers=2
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 330 \
#--pred_len 30 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 300 \
#--pred_len 60 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#e_layers=4
#d_layers=1
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 330 \
#--pred_len 30 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 300 \
#--pred_len 60 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features S \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 1 \
#--dec_in 1 \
#--c_out 1 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#e_layers=3
#d_layers=2
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 360 \
#--pred_len 360 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#e_layers=4
#d_layers=1
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 270 \
#--pred_len 90 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 180 \
#--pred_len 180 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024
#
#python -u run.py \
#--is_training 1 \
#--root_path ./dataset/sst/ \
#--data_path sst_tw.csv \
#--model_id sst_tw \
#--model TransSt \
#--data custom \
#--features M \
#--freq d \
#--des 'Exp' \
#--embed 'timeF' \
#--enc_in 25 \
#--dec_in 25 \
#--c_out 25 \
#--batch_size 32 \
#--seq_len 360 \
#--label_len 360 \
#--pred_len 360 \
#--e_layers $e_layers \
#--d_layers $d_layers \
#--factor 3 \
#--train_epochs 10 \
#--itr 6 \
#--patience 3 \
#--activation 'gelu' \
#--n_heads 8 \
#--moving_avg 45 \
#--d_model 256 \
#--d_ff 1024