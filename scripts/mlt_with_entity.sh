# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=$1
CONTEXT=./
cd ${CONTEXT}
DATA_DIR=./data/test/$2


PRETRAINED_CHECKPOINT=$3


SAVE_DIR=$4
#SAVE_DIR=$1
# cot evaluation
python -m eval.run_eval_mtl_with_entity \
    --data_dir ${DATA_DIR} \
    --save_dir ${SAVE_DIR} \
    --model ${PRETRAINED_CHECKPOINT} \
    --tokenizer ${PRETRAINED_CHECKPOINT} \
    --eval_batch_size $5 \
    --n_shot 0 \
    --use_chat_format \

