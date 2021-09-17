from os import listdir
from os.path import join
import pandas as pd
from util import load_from_bin, save_to_bin
from pprint import pprint

idx_dict = load_from_bin("idx_to_label_dict")
pprint(idx_dict)

train_tiny = load_from_bin("training_tiny")
print(train_tiny.head(3))
print(train_tiny.columns)

test_tiny = load_from_bin("test_tiny")
print(test_tiny.head(3))
print(test_tiny.columns)
