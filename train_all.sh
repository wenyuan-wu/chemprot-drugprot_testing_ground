#!/bin/bash

python drugprot_preprocess.py
python drugprot_preprare_data_none.py
python drugprot_preprare_data_bio.py
python drugprot_preprare_data_sci.py

python lm_train.py
python lm_predict.py

python lm_train_bio.py
python lm_predict_bio.py

python lm_train_sci.py
python lm_predict_sci.py
