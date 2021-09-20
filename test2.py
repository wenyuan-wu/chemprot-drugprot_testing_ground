from util import load_from_bin, check_gpu_mem
from pprint import pprint
from tfdf_train import tfdf_predict, tfdf_train
import tensorflow_decision_forests as tfdf
import logging
import pandas as pd
import numpy as np
from util import load_from_bin, save_to_bin
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )

kg_models = ["TransE", "PairRE"]
models = [
    "bert-base-uncased",
    "allenai/scibert_scivocab_uncased",
    "dmis-lab/biobert-base-cased-v1.1",
]
annotations = ["raw", "sci", "bio"]
datasets = ["development", "test"]

kg_model = "TransE"
lm_model = "bert-base-uncased"
annotation = "raw"
on_tiny = True
model_ann_name = lm_model + "_" + annotation + "_tiny_ft" \
    if on_tiny else lm_model + "_" + annotation + "_ft"
label_name = f"label_tfdf_{kg_model}_{model_ann_name}"
print(label_name)

df = load_from_bin("development_tiny")
y_pred = df[label_name].values
y_true = df["label"].values

print(y_pred[:3])
print(y_true[:3])

print(f1_score(y_true, y_pred, average="micro"))
