from os import listdir
from os.path import join
import pandas as pd
from util import load_from_bin, save_to_bin


train_df = load_from_bin("train_drop_0.85")
train_df = train_df.reset_index()
train_df.rename(columns={'index': "sent_id"}, inplace=True)
# TODO: fix cat issues

train_df.relation = pd.Categorical(train_df.relation)
train_df["label"] = train_df.relation.cat.codes
train_df = train_df.sample(5)
print(train_df.to_string())
label_dict = dict(enumerate(train_df['relation'].cat.categories))
print(label_dict)
save_to_bin(label_dict, "train_drop_0.85_label_dict")

