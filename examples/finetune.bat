export KMER=6
export MODEL_PATH='/mnt/c/Users/m3maf/Documents/DNABERT/models/6-new-12w-0/'
export DATA_PATH='/mnt/c/Users/m3maf/Documents/DNABERT/examples/sample_data/ft/prom-core/3'
export OUTPUT_PATH='/mnt/c/Users/m3maf/Documents/DNABERT/examples/OUTPUT/6mer'

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 75 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-4 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 2000 \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
