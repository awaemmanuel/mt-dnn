#! /bin/sh
python src/mt_dnn/experiments/glue/glue_prepro.py
python src/mt_dnn/embedding_utils/prepro_std.py --model bert-base-uncased --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml --do_lower_case $1
