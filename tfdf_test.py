import tensorflow_decision_forests as tfdf
import logging
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )

logging.info(f'Found TF-DF v{tfdf.__version__}')

dataset_df = pd.read_csv("/tmp/penguins.csv")
print(dataset_df.head(3))

# Name of the label column.
label = "species"

classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)


# Split the dataset into a training and a testing dataset.


def split_dataset(dataset, test_ratio=0.30):
    """Splits a panda dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing.".format(
    len(train_ds_pd), len(test_ds_pd)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label)

# Specify the model.
model_1 = tfdf.keras.RandomForestModel()

# Optionally, add evaluation metrics.
model_1.compile(
    metrics=["accuracy"])

model_1.fit(x=train_ds)

evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

model_1.save("/tmp/my_saved_model")

# print(tfdf.model_plotter.plot_model(model_1, tree_idx=0, max_depth=3))
model_1.summary()

# The input features
print(model_1.make_inspector().features())

# The feature importances
print(model_1.make_inspector().variable_importances())

