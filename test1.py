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
# dev_df = load_from_bin("dev")
# dev_dict = load_from_bin("dev_org")
# pmid = 17380207
# # print(dev_df.index)
# # print(dev_df.iloc[dev_df.index.startswith(pmid)])
# # pprint(dev_dict[pmid])
#
# for k, v in dev_dict.items():
#     if v["pmid"] == pmid:
#         print(f"key: {k}")
#         pprint(v)


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

sent_range = range(21, 38)
# t1_s = 29
# t1_e = 33
# t2_s = 26
# t2_e = 27
# t1_p = "CHEM"
# t2_p = "GENE"
sent = "This an apple tree."
print(len(sent))
new_list = []
ent_dict = {
    "T1": {
        "Start": 29,
        "End": 33,
        "Type": "CHEM"
    },
    "T2": {
        "Start": 26,
        "End": 27,
        "Type": "GENE"
    }
}
srt_list = sorted(ent_dict.items(), key=lambda x: x[1]["Start"], reverse=False)
print(srt_list)
# [('T2', {'Start': 26, 'End': 27, 'Type': 'GENE'}), ('T1', {'Start': 29, 'End': 33, 'Type': 'CHEM'})]
soi = list(sent_range)[0]
idx_1 = srt_list[0][1]["Start"] - soi
char_1 = "<< " if srt_list[0][1]["Type"].startswith("CHEM") else "[[ "
idx_2 = srt_list[0][1]["End"] - soi
char_2 = " >>" if srt_list[0][1]["Type"].startswith("CHEM") else " ]]"
print(idx_1)
print(char_1)
print(idx_2)
print(char_2)
idx_3 = srt_list[1][1]["Start"] - soi
char_3 = "<< " if srt_list[1][1]["Type"].startswith("CHEM") else "[[ "
idx_4 = srt_list[1][1]["End"] - soi
char_4 = " >>" if srt_list[1][1]["Type"].startswith("CHEM") else " ]]"
print(idx_3)
print(char_3)
print(idx_4)
print(char_4)
for idx, char in enumerate(list(sent)):
    print(f"idx: {idx}\tchar: {char}")
    if idx == idx_1:
        char = char_1 + char
    elif idx == idx_2:
        char = char + char_2
    elif idx == idx_3:
        char = char_3 + char
    elif idx == idx_4:
        char = char + char_4
    new_list.append(char)

result = "".join(new_list)
print(result)
print(sent == result)
