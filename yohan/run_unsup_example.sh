#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
    #--model_name_or_path vinai/bertweet-base \

#vinai/bertweet-base
#cardiffnlp/twitter-roberta-base-2021-124m
# 1. 전처리 (필요없는 토큰제거)
# 2. BertTweet 바꿔서 돌린다.
# 3. EvalSet Emoji.

# MODEL_NAME=$1
# MODEL=$2
# MLM=$3

# python train.py \
#     --model_name_or_path ${MODEL_NAME} \
#     --train_file data/emoji_corpus_preprocess.txt \
#     --output_dir result/unsup-${MODEL}-base-10epoch-b128-${MLM} \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 128 \
#     --learning_rate 3e-5 \
#     --max_seq_length 32 \
#     --evaluation_strategy steps \
#     --metric_for_best_model contrastive_loss \
#     --load_best_model_at_end \
#     --eval_steps 125 \
#     --pooler_type cls \
#     --mlp_only_train \
#     --overwrite_output_dir \
#     --temp 0.05 \
#     --do_train \
#     --do_eval \
#     --fp16 \
#     --do_mlm

python train.py \
    --model_name_or_path cardiffnlp/twitter-roberta-base-2021-124m \
    --train_file data/emoji_corpus_preprocess.txt \
    --output_dir result/unsup-timelm-base-10epoch-b128-mlm \
    --num_train_epochs 10 \
    --per_device_train_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model contrastive_loss \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --do_mlm
