
export NSP_MODE=mlm_trelation   # mode=RemeMo
# export NSP_MODE=mlm   # mode=LM

###################### Run ID ###########################
export RUN_ID=debug_pretrain

###################### Corpora File Paths ###########################
export DATA_DIR=data/pretrain
export WIKI_PATH=${DATA_DIR}/enwiki-20221101.json
export BOOKCORPUS_PATH=${DATA_DIR}/bookcorpus.json
export WIKI_TEMPORAL_CONTINUOUS_PATH=${DATA_DIR}/enwiki-20221101_temporal-sentences_special-token-prefix.json

###################### Pretrain Dataset Configuration ######################

export TRAIN_FILES=${WIKI_PATH},${BOOKCORPUS_PATH},${WIKI_TEMPORAL_CONTINUOUS_PATH}
export DATA_USED=wiki_books_twikiCon

###################### GPU & Batch-size Configuration  ######################
########  T5  #######
export BATCH_SIZE=2048
export ADAM_EPSILON=1e-6

export OPTIMIZER=adafactor
# export OPTIMIZER=adamw_torch

# export LR_SCHEDULER=linear
export LR_SCHEDULER=cosine

#  export LEARNING_RATE=1e-3   # t5-small
#  export LEARNING_RATE=5e-4   # t5-base
 export LEARNING_RATE=3e-4   # t5-large

# export BATCH_SIZE_PER_GPU=90   # t5_v1.1-small model, on A6000-48G, with --bf16, with MAX_NUM_CLS=10
# export BATCH_SIZE_PER_GPU=128  # t5_small model, on A6000-48G, with --bf16, with MAX_NUM_CLS=10
export BATCH_SIZE_PER_GPU=32   # t5_small model, debug
# export BATCH_SIZE_PER_GPU=16   # t5_large model, on A6000-48G, with --bf16, with MAX_NUM_CLS=10

export GPU_IDX="0"
export NUM_OF_GPUS=2
# export NUM_OF_GPUS=$(nvidia-smi --list-gpus | wc -l)
export CUDA_VISIBLE_DEVICES=${GPU_IDX}

export GRADIENT_ACCUMULATION_STEPS=$(($(( BATCH_SIZE / BATCH_SIZE_PER_GPU))/NUM_OF_GPUS))

###################### Which Pretrained-LM  ######################
# export PLM_DIR="../pretrained_models/"
export PLM_DIR="google"
export MODEL_NAME_OR_PATH="t5-v1_1-base"

###################### Logging Configuration  ######################
export WANDB_PROJECT=RemeMo-pretrain

 # {'wandb', 'none'}
# export REPORT_TO=none
export REPORT_TO=wandb

###################### Start Training  ######################
export RUN_NAME=${MODEL_NAME_OR_PATH}.${DATA_USED}.${NSP_MODE}.${OPTIMIZER}.${LR_SCHEDULER}.run-${RUN_ID}

# python -m torch.distributed.launch --nproc_per_node ${NUM_OF_GPUS} run_seq2seq.py \
python run_seq2seq.py \
--train_files ${TRAIN_FILES} \
--model_name_or_path ${PLM_DIR}${MODEL_NAME_OR_PATH}  \
--max_source_length 512 \
--per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
--max_steps 8000 \
--learning_rate ${LEARNING_RATE} \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--adam_beta1 0.9 \
--adam_beta2 0.98 \
--adam_epsilon ${ADAM_EPSILON} \
--report_to ${REPORT_TO} \
--run_name ${RUN_NAME} \
--nsp_mode ${NSP_MODE} \
--do_train \
--remove_unused_columns false \
--logging_steps 2 \
--logging_first_step true \
--output_dir log/${RUN_NAME} \
--save_strategy steps \
--save_steps 500 \
--max_grad_norm 1.0 \
--optim ${OPTIMIZER} \
--lr_scheduler_type ${LR_SCHEDULER} \
2>&1 | tee log/${RUN_NAME}.txt

# --bf16