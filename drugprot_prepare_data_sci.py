import logging
import pandas as pd
from util import save_to_bin, load_from_bin
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )

df_train = pd.DataFrame.from_dict(load_from_bin("train_org"), orient="index")
df_train = df_train[['text_scibert', 'relation']]
# drop none labels
df_train = df_train.drop(df_train[df_train["relation"] == "NONE"].index)
df_train.relation = pd.Categorical(df_train.relation)
df_train["label"] = df_train.relation.cat.codes
idx_to_label_dict = dict(enumerate(df_train['relation'].cat.categories))
# print(idx_to_label_dict)
label_to_idx_dict = {v: k for k, v in idx_to_label_dict.items()}
print(label_to_idx_dict)
save_to_bin(label_to_idx_dict, "label_to_idx_dict_sci")
save_to_bin(idx_to_label_dict, "idx_to_label_dict_sci")

print(df_train.sample(10, random_state=1024).to_string())
# print(df_train.loc[df_train.relation != "NONE"].sample(5, random_state=1024))
print(df_train["relation"].value_counts())

# plot data
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (10, 5)

# Plot the number of tokens of each length.
sns.countplot(x="relation", data=df_train.loc[df_train.relation != "NONE"])

plt.title('Class Distribution')
plt.xlabel('Category')
plt.ylabel('# of Training Samples')
# plt.show()

save_to_bin(df_train, "train_sci")
df_train_tiny = df_train.drop(df_train.sample(frac=.999, random_state=1024).index)
save_to_bin(df_train_tiny, "train_tiny_sci")

logging.info("development set")
df_dev = pd.DataFrame.from_dict(load_from_bin("dev_org"), orient="index")
df_dev = df_dev[['text_scibert', 'relation']]
df_dev = df_dev.drop(df_dev[df_dev["relation"] == "NONE"].index)
df_dev["label"] = df_dev["relation"].map(label_to_idx_dict)

print(df_dev.sample(10, random_state=1024).to_string())
# print(df_dev.loc[df_dev.relation != "NONE"].sample(5))
print(df_dev["relation"].value_counts())

sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (10, 5)

# Plot the number of tokens of each length.
sns.countplot(x="relation", data=df_dev.loc[df_dev.relation != "NONE"])

plt.title('Class Distribution')
plt.xlabel('Category')
plt.ylabel('# of Training Samples')
# plt.show()

save_to_bin(df_dev, "dev_sci")
df_dev_tiny = df_dev.drop(df_dev.sample(frac=.999, random_state=1024).index)
save_to_bin(df_dev_tiny, "dev_tiny_sci")


# test set
# logging.info("test set")
# df_test = pd.DataFrame.from_dict(load_from_bin("test_org"), orient="index")
# df_test = df_test[["test_scibert", "relation"]]
# df_test["label"] = [0] * df_test.shape[0]
# print(df_test.sample(5))
# print(df_test["relation"].value_counts())
#
# save_to_bin(df_test, "test")
# df_test_tiny = df_test.drop(df_test.sample(frac=.999, random_state=1024).index)
# save_to_bin(df_test_tiny, "test_tiny")