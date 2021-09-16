#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://#opensource.org/licenses/BSD-3-Clause

export DATA_DIR=outputs
ACTION=${1:-none}
dataset="grail"

if [ "$ACTION" = "train" ]; then
    exp_id=$2

    exp_prefix="exps/rank_${dataset}_${exp_id}/"

    mkdir ${exp_prefix}
    cp scripts/run_ranker.sh "${exp_prefix}run_ranker.sh"
    git rev-parse HEAD > "${exp_prefix}commitid.log"

    python -u run_ranker.py \
        --dataset ${dataset} \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --do_train \
        --do_eval \
        --disable_tqdm \
        --train_file $DATA_DIR/${dataset}_train_candidates-ranking.jsonline \
        --predict_file $DATA_DIR/${dataset}_dev_candidates-ranking.jsonline \
        --learning_rate 3e-5 \
        --evaluate_during_training \
        --num_train_epochs 3 \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --logging_steps 1000 \
        --eval_steps 5000 \
        --save_steps 5000 \
        --warmup_ratio 0.1 \
        --output_dir "${exp_prefix}output" \
        --training_curriculum mixbootstrap \
        --bootstrapping_start 2 \
        --bootstrapping_ticks 3 \
        --num_contrast_sample 96 \
        --per_gpu_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --per_gpu_eval_batch_size 128 | tee "${exp_prefix}log.txt"
elif [ "$ACTION" = "eval" -o "$ACTION" = "predict" ]; then
    model=$2
    split=${3:-dev}
    python -u run_ranker.py \
        --dataset ${dataset} \
        --model_type bert \
        --model_name_or_path ${model} \
        --do_lower_case \
        --do_eval \
        --predict_file $DATA_DIR/${dataset}_${split}_candidates-ranking.jsonline \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --output_dir  results/ranker/${dataset}_${split} \
        --per_gpu_eval_batch_size 128
    cp results/ranker/${dataset}_${split}/candidate_logits.bin misc/grail_${split}_candidate_logits.bin
    cp results/ranker/${dataset}_${split}/predictions.txt misc/grail_${split}_ranker_results.txt
else
    echo "train or eval"
fi