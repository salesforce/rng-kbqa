#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://#opensource.org/licenses/BSD-3-Clause

export DATA_DIR=outputs
export condition_id="noncan"
ACTION=${1:-none}

if [ "$ACTION" = "train" ]; then
    dataset=$2
    exp_id=$3
    exp_prefix="exps/${dataset}_${condition_id}_${exp_id}/"

    mkdir ${exp_prefix}
    cp scripts/train_ranker_nonbootstrap.sh "${exp_prefix}train_ranker_nonbootstrap.sh"
    git rev-parse HEAD > "${exp_prefix}commitid.log"

    if [ "$dataset" = "grail" ]; then
        python -u run_ranker.py \
            --dataset ${dataset} \
            --model_type bert \
            --model_name_or_path bert-base-uncased \
            --do_lower_case \
            --do_train \
            --do_eval \
            --disable_tqdm \
            --train_file $DATA_DIR/${dataset}_train_candidates-${condition_id}.jsonline \
            --predict_file $DATA_DIR/${dataset}_dev_candidates-${condition_id}.jsonline \
            --learning_rate 3e-5 \
            --num_train_epochs 3 \
            --overwrite_output_dir \
            --max_seq_length 96 \
            --logging_steps 1000 \
            --eval_steps 50000 \
            --save_steps 10000 \
            --warmup_ratio 0.1 \
            --output_dir "${exp_prefix}output" \
            --num_contrast_sample 96 \
            --per_gpu_train_batch_size 1 \
            --gradient_accumulation_steps 2 \
            --per_gpu_eval_batch_size 128 | tee "${exp_prefix}log.txt"
    else
        echo "invalid dataset"
    fi

elif [ "$ACTION" = "eval" ]; then
    model=$2
    dataset=$3
    split=${4:-dev}
    python -u run_ranker.py \
        --model_type bert \
        --model_name_or_path ${model} \
        --do_eval \
        --predict_file $DATA_DIR/${dataset}_${split}_candidates-${condition_id}.jsonline \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --output_dir  results/${dataset} \
        --per_gpu_eval_batch_size 128
else
    echo "train or eval"
fi
