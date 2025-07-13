export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CONTEXT=./
PRETRAINED_PATH=$1
CODE_FILE=${CONTEXT}/finetune_llm.py
DS_CONFIG=${CONTEXT}/ds_configs/stage3_no_offloading_accelerate.conf
TRAIN_FILE=./training_data/$2
DEV_FILE=./training_data/$3
OUTPUT_DIR=${CONTEXT}/output/checkpoint/$4
MODEL_SIZE=13B
NUM_GPUS=8
NUM_MECHINES=1
BATCH_SIZE_PER_GPU=1
DEV_BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines ${NUM_MECHINES} \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 29533 \
    --deepspeed_config_file ${DS_CONFIG} \
    ${CODE_FILE} \
    --model_name_or_path ${PRETRAINED_PATH} \
    --use_flash_attn \
    --tokenizer_name ${PRETRAINED_PATH} \
    --use_slow_tokenizer \
    --train_file ${TRAIN_FILE} \
    --dev_file ${DEV_FILE} \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_dev_batch_size $DEV_BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --adamw_beta1 0.9 \
    --adamw_beta2 0.999 \
    --adamw_eps 1e-8 \
    --num_train_epochs 3 \
    --output_dir ${OUTPUT_DIR} \
    --with_tracking \
    --report_to all \
    --logging_steps 500 \
    --checkpointing_steps epoch \
    --evaluation_steps epoch