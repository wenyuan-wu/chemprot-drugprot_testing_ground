#!/bin/bash

python drugprot_preprocess.py
python drugprot_lm_finetune.py
python drugprot_lm_predict.py
python drugprot_lm_evaluate.py

python pykeen_prepare_data.py
python pykeen_train.py

python tfdf_train.py
