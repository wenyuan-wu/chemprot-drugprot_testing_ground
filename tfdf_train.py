import tensorflow_decision_forests as tfdf
import logging
import pandas as pd
import numpy as np
from util import load_from_bin, save_to_bin
import tensorflow as tf

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )

logging.info(f'Found TF-DF v{tfdf.__version__}')

df = load_from_bin("train_sci_lm_kg")
# print(df.head(3))
# print(df.columns)
df = df[["con_embd", "label"]]
print(df.columns)
# Name of the label column.
label = "label"

x = np.asarray(list(df.con_embd.values)).astype(np.float32)
y = np.asarray(list(df.label.values)).astype(np.int64)

# Specify the model.
model_1 = tfdf.keras.RandomForestModel()
# #
# Optionally, add evaluation metrics.
model_1.compile(
    metrics=["accuracy"])

model_1.fit(x, y)
#
# evaluation = model_1.evaluate(x_test, y_test, return_dict=True)
# print()
# #
# for name, value in evaluation.items():
#     print(f"{name}: {value:.4f}")

model_1.save("model/tfdf_raw_tiny")

df = load_from_bin("test_sci_lm_kg")
x_test = np.asarray(list(df.con_embd.values)).astype(np.float32)

pred = model_1.predict(x_test)
predicted_labels = np.argmax(pred, axis=1)
print("="*20)
print(predicted_labels.shape)

idx_to_label_dict = load_from_bin("idx_to_label_dict_raw")
print(df.columns)
df["label_pred"] = predicted_labels
df["relation_pred"] = df["label_pred"].map(idx_to_label_dict)
print(df.columns)
save_to_bin(df, "test_sci_lm_kg_dp")
