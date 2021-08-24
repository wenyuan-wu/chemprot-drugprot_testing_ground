from os import listdir
from os.path import join
import pandas as pd
from util import load_from_bin, save_to_bin
from pprint import pprint

# df_test = load_from_bin("test_tiny_pred")
# # print(df_test)
#
# dict_test = load_from_bin("test_org")
# # pprint(dict_test)
# data_dict = {}
# for idx, row in df_test.iterrows():
#     print(idx)
#     print(row)
#     pprint(dict_test[idx])
#
#     print("=" * 25)
#     break
dev_df = load_from_bin("dev")
dev_dict = load_from_bin("dev_org")
pmid = 17380207
# print(dev_df.index)
# print(dev_df.iloc[dev_df.index.startswith(pmid)])
# pprint(dev_dict[pmid])

for k, v in dev_dict.items():
    if v["pmid"] == pmid:
        print(f"key: {k}")
        pprint(v)


# print(dev_df)
# dev_dict = load_from_bin("dev_org")
# counter = 0
#
# for idx, row in dev_df.iterrows():
#     print(idx)
#     print(row)
#     counter += 1
#     pprint(dev_dict[idx])
#     print("=" * 40)
#     if counter > 5:
#         break
#
