export KMER=3
export MODEL_PATH=/mnt/c/Users/m3maf/Documents/DNABERT/examples/OUTPUT/3mer
export DATA_PATH=/mnt/c/Users/m3maf/Documents/DNABERT/examples/sample_data/ft/prom-core/3
export PREDICTION_PATH=/mnt/c/Users/m3maf/Documents/DNABERT/examples/OUTPUT/pred

python run_finetune.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 75 \
    --per_gpu_pred_batch_size=128   \
    --output_dir $MODEL_PATH \
    --predict_dir $PREDICTION_PATH \
    --n_process 48
