#!/bin/bash

python drugprot_preprocess.py
python drugprot_prepare_data.py

python lm_train.py
python lm_predict.py

python lm_train_bio.py
python lm_predict_bio.py

python lm_train_sci.py
python lm_predict_sci.py
