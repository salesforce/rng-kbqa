export DATA_DIR=outputs
ACTION=${1:-none}
dataset="webqsp"

if [ "$ACTION" = "train" ]; then
    exp_id=$2

    exp_prefix="exps/gen_${dataset}_${exp_id}/"

    mkdir ${exp_prefix}
    cp scripts/run_gen.sh "${exp_prefix}run_gen.sh"
    git rev-parse HEAD > "${exp_prefix}commitid.log"

    python -u run_generator.py \
        --dataset ${dataset} \
        --model_type t5 \
        --model_name_or_path t5-base \
        --do_lower_case \
        --do_train \
        --do_eval \
        --train_file ${DATA_DIR}/${dataset}_ptrain_gen.json \
        --predict_file ${DATA_DIR}/${dataset}_pdev_gen.json \
        --learning_rate 3e-5 \
        --evaluate_during_training \
        --num_train_epochs 20 \
        --overwrite_output_dir \
        --logging_steps 348 \
        --eval_steps 348 \
        --save_steps 348 \
        --warmup_ratio 0.1 \
        --output_dir "${exp_prefix}output" \
        --eval_beams 10 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 8 | tee "${exp_prefix}log.txt"

elif [ "$ACTION" = "eval" -o "$ACTION" = "predict" ]; then
    model=$2
    split=${3:-dev}
    python -u run_generator.py \
        --dataset ${dataset} \
        --model_type t5 \
        --model_name_or_path ${model} \
        --eval_beams 10 \
        --do_lower_case \
        --do_eval \
        --predict_file ${DATA_DIR}/${dataset}_${split}_gen.json \
        --overwrite_output_dir \
        --output_dir  results/gen/${dataset}_${split} \
        --per_device_eval_batch_size 8 
    cp results/gen/${dataset}_${split}/top_k_predictions.json misc/${dataset}_${split}_topk_generations.json
else
    echo "train or eval"
fi
