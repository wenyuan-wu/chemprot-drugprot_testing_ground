import logging
import pandas as pd
from util import save_to_bin, load_from_bin
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def prepare_data(dataset="train", annotation="raw", drop_none=False, frac=0.99, random_state=1024):
    """
    Prepare data for training from preprocessed data dictionary
    :param dataset: type of the dataset, ["train", "dev", "test"]
    :param annotation: annotation style, ["raw", "sci", "bio"]
    :param drop_none: boolean, if True then drop NONE relation in dataframe
    :param frac: fraction to create tiny dataset for testing
    :param random_state: random state for dropping
    :return: None, prepared files will be saved accordingly
    """
    data_name = dataset + "_org"
    df = pd.DataFrame.from_dict(load_from_bin(data_name), orient="index")
    ann_style = "text_" + annotation
    df = df[[ann_style, "relation", "pmid", "Arg1", "Arg2", "Ent1", "Ent2"]]
    if drop_none:
        # drop sentences with NONE relation
        df = df.drop(df[df["relation"] == "NONE"].index)
    if dataset == "train":
        # convert relation into numerical categories
        df.relation = pd.Categorical(df.relation)
        df["label"] = df.relation.cat.codes
        idx_to_label_dict = dict(enumerate(df['relation'].cat.categories))
        label_to_idx_dict = {v: k for k, v in idx_to_label_dict.items()}
        # create tiny dataset for quick testing purpose
        df_tiny = df.drop(df.sample(frac=frac, random_state=random_state).index)
        # saving files as binary format
        save_to_bin(label_to_idx_dict, f"label_to_idx_dict_{annotation}")
        save_to_bin(idx_to_label_dict, f"idx_to_label_dict_{annotation}")
        save_to_bin(df, f"{dataset}_{annotation}")
        save_to_bin(df_tiny, f"{dataset}_{annotation}_tiny")
    else:
        # mapping relations into numerical categories from loaded mapping dictionary
        label_to_idx_dict = load_from_bin(f"label_to_idx_dict_{annotation}")
        df["label"] = df["relation"].map(label_to_idx_dict)
        df_tiny = df.drop(df.sample(frac=frac, random_state=random_state).index)
        save_to_bin(df, f"{dataset}_{annotation}")
        save_to_bin(df_tiny, f"{dataset}_{annotation}_tiny")


def plot_label_dist(dataset="train", drop_none=False):
    """
    Plot the distribution information of the dataset
    :param dataset: training or development set
    :param drop_none: boolean, if True then drop NONE relation in dataframe
    :return: matplotlib.pyplot instance
    """
    data_name = dataset + "_org"
    df = pd.DataFrame.from_dict(load_from_bin(data_name), orient="index")
    if drop_none:
        df = df.drop(df[df["relation"] == "NONE"].index)
    logging.info(f"distribution info:\n{df['relation'].value_counts()}")
    # plot data
    sns.set(style='darkgrid')
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (50, 25)
    # Plot the number of tokens of each length.
    sns.countplot(x="relation", data=df)
    plt.title('Relation Distribution')
    plt.xlabel('Relation')
    plt.ylabel('# of Training Samples')
    return plt


def main():
    # prepare data for next step
    for dataset in ["train", "dev", "test"]:
        for ann in ["raw", "sci", "bio"]:
            prepare_data(dataset, ann)

    # save plotted images of label distribution information
    plt_train = plot_label_dist("train")
    plt_train.savefig("info/train_none.png")
    plt_train = plot_label_dist("train", drop_none=True)
    plt_train.savefig("info/train.png")
    plt_dev = plot_label_dist("dev")
    plt_dev.savefig("info/dev_none.png")
    plt_dev = plot_label_dist("dev", drop_none=True)
    plt_dev.savefig("info/dev.png")


if __name__ == '__main__':
    main()
