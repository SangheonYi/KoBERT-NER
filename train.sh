rm -rf my_data/cache* model
python3 main.py --model_type koelectra-base --do_train --data_dir ./my_data --save_steps 10 --num_train_epochs 10 --train_batch_size 64 --eval_batch_size 64 --max_seq_len 100
