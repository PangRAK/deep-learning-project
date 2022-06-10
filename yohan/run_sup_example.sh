#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=4

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# BerTweet / Stance / Near
python train.py \
    --model_name_or_path ./vinai/bertweet-base \
    --train_file /home/rak/git/deep-learning-project/sangrak/tweeteval_stance_random.csv \
    --output_dir result/sentiment-sup-simcse-bertweet-base \
    --num_train_epochs 10 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"




# python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
#     --model_name_or_path bert-base-uncased \
#     --train_file data/nli_for_simcse.csv \
#     --output_dir result/my-sup-simcse-bert-base-uncased \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 128 \
#     --learning_rate 5e-5 \
#     --max_seq_length 32 \
#     --evaluation_strategy steps \
#     --metric_for_best_model stsb_spearman \
#     --load_best_model_at_end \
#     --eval_steps 125 \
#     --pooler_type cls \
#     --overwrite_output_dir \
#     --temp 0.05 \
#     --do_train \
#     --do_eval \
#     --fp16 \
#     "$@"

