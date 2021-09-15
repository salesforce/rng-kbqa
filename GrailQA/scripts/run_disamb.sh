export DATA_DIR=outputs
ACTION=${1:-none}
dataset="grail"
if [ "$ACTION" = "train" ]; then
    exp_id=$2

    exp_prefix="exps/disamb_${dataset}_${exp_id}/"

    mkdir ${exp_prefix}
    cp scripts/run_disamb.sh "${exp_prefix}run_disamb.sh"
    git rev-parse HEAD > "${exp_prefix}commitid.log"

    python -u run_disamb.py \
        --dataset ${dataset} \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --do_train \
        --do_eval \
        --disable_tqdm \
        --train_file $DATA_DIR/${dataset}_train_entities.json \
        --predict_file $DATA_DIR/${dataset}_dev_entities.json \
        --learning_rate 1e-5 \
        --evaluate_during_training \
        --num_train_epochs 2 \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --logging_steps 200 \
        --eval_steps 500 \
        --save_steps 1000 \
        --warmup_ratio 0.1 \
        --output_dir "${exp_prefix}output" \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 128 | tee "${exp_prefix}log.txt"

elif [ "$ACTION" = "eval" ]; then
    model=$2
    split=${3:-dev}
    python -u run_disamb.py \
        --dataset ${dataset} \
        --model_type bert \
        --model_name_or_path ${model} \
        --do_lower_case \
        --do_eval \
        --predict_file $DATA_DIR/${dataset}_${split}_entities.json \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --output_dir  results/disamb/${dataset} \
        --per_gpu_eval_batch_size 128

elif [ "$ACTION" = "predict" ]; then
    model=$2
    split=${3:-dev}
    python -u run_disamb.py \
        --dataset ${dataset} \
        --model_type bert \
        --model_name_or_path ${model} \
        --do_lower_case \
        --do_eval \
        --do_predict \
        --predict_file $DATA_DIR/${dataset}_${split}_entities.json \
        --overwrite_output_dir \
        --max_seq_length 96 \
        --output_dir  results/disamb/${dataset}_${split} \
        --per_gpu_eval_batch_size 128
    cp results/disamb/${dataset}_${split}/predictions.json misc/${dataset}_${split}_entity_linking.json
else
    echo "train or eval"
fi