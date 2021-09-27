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
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )

my_pykeen_model = torch.load('model/PairRE/trained_model.pkl')

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in my_pykeen_model.state_dict():
    print(param_tensor, "\t", my_pykeen_model.state_dict()[param_tensor].size())

entity_embedding_tensor = my_pykeen_model.state_dict()["entity_representations.0._embeddings.weight"].cpu().detach().numpy()
logging.info(f"sheep: {entity_embedding_tensor.shape}")
embd_shape = entity_embedding_tensor.shape[1]
print(embd_shape)
