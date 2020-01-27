#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=" "
python3 -m torch.distributed.launch --nproc_per_node=8 ./transformers/examples/run_squad.py \
    --model_type bert \
    --no_cuda \
    --local_rank -1 \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file custom_train.json \
    --predict_file dev.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./models/wwm_uncased_finetuned_squad/ \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3   \