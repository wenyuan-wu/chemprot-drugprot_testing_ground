from os import listdir
from os.path import join
import pandas as pd
from util import load_from_bin, save_to_bin
from pprint import pprint

df_test = load_from_bin("test_tiny_pred")
print(df_test)

dict_test = load_from_bin("test")
# pprint(dict_test)

for idx, row in df_test.iterrows():
    print(idx)
    print(row)
    print(dict_test[idx])
    print("=" * 25)
    break
