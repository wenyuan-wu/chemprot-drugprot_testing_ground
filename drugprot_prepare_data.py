import logging
import pandas as pd
from util import save_to_bin, load_from_bin
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )

df_train = pd.DataFrame.from_dict(load_from_bin("train"), orient="index")
df_train = df_train[['text_raw', 'relation']]

print(df_train.sample(5, random_state=1024))
print(df_train.loc[df_train.relation != "NONE"].sample(5, random_state=1024))
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

# randomly drop 85% of NONE labels
df_train = df_train.drop(df_train[df_train["relation"] == "NONE"].sample(frac=.85, random_state=1024).index)
print(df_train["relation"].value_counts())
save_to_bin(df_train, "train_drop_0.85")

logging.info("development set")
df_dev = pd.DataFrame.from_dict(load_from_bin("dev"), orient="index")
df_dev = df_dev[['text_raw', 'relation']]

print(df_dev.sample(5))
print(df_dev.loc[df_dev.relation != "NONE"].sample(5))
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

# randomly drop 85% of NONE labels
df_dev = df_dev.drop(df_dev[df_dev["relation"] == "NONE"].sample(frac=.85, random_state=1024).index)
print(df_dev["relation"].value_counts())
save_to_bin(df_dev, "dev_drop_0.85")
