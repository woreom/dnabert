export KMER=3
export MODEL_PATH='/mnt/d/M3/Projects/BCB/DNABERT/examples/OUTPUT/pre/3mer/checkpoint-32000/'
export DATA_PATH='/mnt/d/M3/Projects/BCB/DNABERT/examples/sample_data/ft/prom-core/3'
export OUTPUT_PATH='/mnt/d/M3/Projects/BCB/DNABERT/examples/OUTPUT/fit/3mer/'

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 23 \
    --per_gpu_eval_batch_size=64   \
    --per_gpu_train_batch_size=64   \
    --learning_rate 1e-1 \
    --num_train_epochs 5.0 \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 10000 \
    --warmup_percent 0.2 \
    --hidden_dropout_prob 0.5 \
    --overwrite_output \
    --weight_decay 0.0001 \
    --n_process 24