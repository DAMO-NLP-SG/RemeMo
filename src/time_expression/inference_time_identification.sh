
export ROOT_DIR="../.."
export CURR_INPUT=""
export CURR_OUTPUT=""

python token-classification/run_ner.py \
	--model_name_or_path ${ROOT_DIR}/model_checkpoints/time_expression/roberta_for_time_identification \
	--train_file ${ROOT_DIR}/data/time_expression/ner_task/train.json \
	--validation_file ${ROOT_DIR}/data/time_expression/ner_task/val.json \
	--text_column_name tokens \
	--label_column_name ner_tags \
	--per_device_eval_batch_size 128  \
	--output_dir ${ROOT_DIR}/model_checkpoints/zzz/ \
	--test_file ${CURR_INPUT} \
	--predict_output ${CURR_OUTPUT} \
	--report_to none \
	--run_name predict-timeNER \
	--max_seq_length 512  \
	--do_predict
