import logging
import pandas as pd
from util import load_from_bin
from os.path import join

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def create_kg_data(dataframe: pd.DataFrame, contains_none=True) -> list:
    """
    Prepare data for pykeen to train kg model
    :param dataframe: dataframe to load for process
    :param contains_none: if set contains none value
    :return: list of tuples
    """
    return list(zip(dataframe["Ent1"].values,
                    dataframe["relation"].values,
                    dataframe["Ent2"].values))


def write_to_file(kg_list: list, file_name: str) -> None:
    """
    Function to write processed list into tsv file
    :param kg_list: list of node and relation tuples
    :param file_name: string to indicate file name for saving
    :return: None
    """
    file_path = join("data", "drugprot_preprocessed", "kg", file_name)
    with open(file_path, "w") as outfile:
        for row in kg_list:
            try:
                outfile.write("\t".join(row))
                outfile.write("\n")
            except TypeError:
                logging.error(f"Unexpected values found: {row}")
    logging.info(f"file saved in {file_path}")


def main():
    train_list = create_kg_data(load_from_bin("training"))
    write_to_file(train_list, "drugprot_kg_train.tsv")
    dev_list = create_kg_data(load_from_bin("development"))
    write_to_file(dev_list, "drugprot_kg_dev.tsv")
    combi_list = train_list + dev_list
    write_to_file(combi_list, "drugprot_kg_train_combi.tsv")


if __name__ == '__main__':
    main()
