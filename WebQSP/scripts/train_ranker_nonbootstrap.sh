#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://#opensource.org/licenses/BSD-3-Clause

export DATA_DIR=outputs
ACTION=${1:-none}
dataset="webqsp"

if [ "$ACTION" = "train" ]; then
    exp_id=$2

    exp_prefix="exps/${dataset}_${exp_id}/"

    mkdir ${exp_prefix}
    cp scripts/run_ranker_nonbootsrap.sh "${exp_prefix}run_ranker_nonbootsrap.sh"
    git rev-parse HEAD > "${exp_prefix}commitid.log"

    python -u run_ranker.py \
        --dataset ${dataset} \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --do_train \
        --do_eval \
        --disable_tqdm \
        --train_file $DATA_DIR/${dataset}_ptrain_candidates-ranking.json \
        --predict_file $DATA_DIR/${dataset}_pdev_candidates-ranking.json \
        --learning_rate 3e-5 \
        --num_train_epochs 10 \
        --gradient_accumulation_steps 2 \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --logging_steps 371 \
        --eval_steps 148400 \
        --save_steps 74200 \
        --warmup_ratio 0.1 \
        --output_dir "${exp_prefix}output" \
        --num_contrast_sample 96 \
        --per_gpu_train_batch_size 1 \
        --per_gpu_eval_batch_size 128 | tee "${exp_prefix}log.txt"
elif [ "$ACTION" = "eval" -o "$ACTION" = "predict" ]; then
    model=$2
    split=${3:-test}
    python -u run_ranker.py \
        --dataset ${dataset} \
        --model_type bert \
        --model_name_or_path ${model} \
        --do_lower_case \
        --do_eval \
        --predict_file $DATA_DIR/${dataset}_${split}_candidates-ranking.json \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --output_dir  results/ranker/${dataset}_${split} \
        --per_gpu_eval_batch_size 128
    cp results/ranker/${dataset}_${split}/candidate_logits.bin misc/${dataset}_${split}_candidate_logits.bin
    cp results/ranker/${dataset}_${split}/predictions.txt misc/${dataset}_${split}_ranker_results.txt
else
    echo "train or eval"
fi
