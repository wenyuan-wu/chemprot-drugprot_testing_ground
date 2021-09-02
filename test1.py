from os import listdir
from os.path import join
import pandas as pd
from util import load_from_bin, save_to_bin
from pprint import pprint

sent = "This an apple tree and that car is fast."
sent_range = range(21, 38)

print(len(sent))
new_list = []

ent_dict_w = {
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

ent_dict_wo = {
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
idx_2 = srt_list[0][1]["End"] - soi
idx_range_1 = list(range(idx_1 + 1, idx_2 + 1))

char_1 = "$CHMICAL#" if srt_list[0][1]["Type"].startswith("CHEM") else "$GENE#"

print(idx_1)
print(char_1)
print(idx_range_1)

idx_3 = srt_list[1][1]["Start"] - soi
char_2 = "$CHMICAL#" if srt_list[1][1]["Type"].startswith("CHEM") else "$GENE#"
idx_4 = srt_list[1][1]["End"] - soi
idx_range_2 = list(range(idx_3 + 1, idx_4 + 1))
print(idx_3)
print(char_2)
print(idx_range_2)

for idx, char in enumerate(list(sent)):
    print(f"idx: {idx}\tchar: {char}")
    if idx == idx_1:
        char = char_1
    elif idx in idx_range_1:
        char = ""
    elif idx == idx_3:
        char = char_2
    elif idx in idx_range_2:
        char = ""
    new_list.append(char)

result = "".join(new_list)
print(result)
print(sent == result)
