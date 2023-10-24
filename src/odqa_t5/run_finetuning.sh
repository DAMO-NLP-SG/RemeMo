export TRUNCATION_LEN=1500  # 1500, 1000 or 500
export USE_SQUAD_V2=true # set to true if the dataset contains unanswerable questions, otherwise false

export DATASET="tsqa_hard"
export HALF_NUM_TRAIN_EXAMPLES=7340 # (number_of_training_instances / 2)

########################### Dataset Paths ###########################

export DATASET_PATH=data/finetune/${DATASET}
export TRAIN_FILE=${DATASET_PATH}/train.truncate-${TRUNCATION_LEN}.json
export VAL_FILE=${DATASET_PATH}/dev.truncate-${TRUNCATION_LEN}.json
export TEST_FILE=${DATASET_PATH}/test.truncate-${TRUNCATION_LEN}.json

########################### Model Path ###########################

export MODEL="rememo-base"
export MODEL_PATH=DAMO-NLP-SG/${MODEL}

# export MODEL="t5-v1_1-base"
# export MODEL_PATH=google/${MODEL}

########################### wandb ###########################
export WANDB_PROJECT=RemeMo-finetune

# {'wandb', 'none'}
export REPORT_TO=none

########################### GPU, Batch-Size, Learning-Rate ###########################
for BATCH_SIZE_PER_GPU in 8 16
do
for LR in 3e-5 1e-4 3e-4
do

export NUM_TRAIN_EPOCHS=10

export GRADIENT_ACCUMULATION_STEPS=2

export EVAL_STEPS=$((HALF_NUM_TRAIN_EXAMPLES / BATCH_SIZE_PER_GPU))

export FINAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS))

#######################################################################################

export RUN_NAME=${MODEL}.${DATASET}.truncate-${TRUNCATION_LEN}.lr-${LR}.bsz-${FINAL_BATCH_SIZE}.epoch-${NUM_TRAIN_EPOCHS}

python run_seq2seq_qa.py \
  --model_name_or_path ${MODEL_PATH} \
  --train_file ${TRAIN_FILE} \
  --validation_file ${VAL_FILE} \
  --test_file ${TEST_FILE} \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --learning_rate ${LR} \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --max_seq_length 512 \
  --output_dir log/${RUN_NAME} \
  --report_to ${REPORT_TO} \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps ${EVAL_STEPS} \
  --save_steps ${EVAL_STEPS} \
  --logging_first_step true \
  --load_best_model_at_end true \
  --metric_for_best_model eval_f1 \
  --greater_is_better true \
  --save_total_limit 2 \
  --overwrite_output_dir true \
  --remove_unused_columns true \
  --predict_with_generate \
  --run_name ${RUN_NAME} \
  --version_2_with_negative ${USE_SQUAD_V2} \
  2>&1 | tee log/${RUN_NAME}.txt

done
done


