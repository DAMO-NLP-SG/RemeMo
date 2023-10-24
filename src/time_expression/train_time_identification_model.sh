python token-classification/run_ner.py \
    --model_name_or_path roberta-large \
    --train_file ../../data/time_expression/ner_task/train.json \
    --validation_file ../../data/time_expression/ner_task/val.json \
    --text_column_name tokens \
    --label_column_name ner_tags \
    --output_dir ../../model_checkpoints/my_own_roberta_for_time_identification \
    --do_train \
    --do_eval