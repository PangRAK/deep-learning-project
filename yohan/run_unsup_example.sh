#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
# cardiffnlp/twitter-roberta-base-mar2022
# bert-base-uncased
# roberta-base

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file /home/uj-user/deep-learning-project/yohan/data/emoji_corpus_preprocess.txt \
    --output_dir result/roberta-base2 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_eval \
    --fp16 \
    "$@"

#vinai/bertweet-base
# 1. 전처리 (필요없는 토큰제거)
# 2. BertTweet 바꿔서 돌린다.
# 3. EvalSet Emoji.