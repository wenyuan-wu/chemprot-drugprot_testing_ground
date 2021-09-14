import logging
import pandas as pd
from util import load_from_bin
from os.path import join

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def create_kg_data(dataframe: pd.DataFrame, contains_none=True):
    return list(zip(dataframe["Ent1"].values,
                    dataframe["relation"].values,
                    dataframe["Ent2"].values))


def write_to_file(kg_list, file_name):
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
    df_train = pd.DataFrame.from_dict(load_from_bin("train_org"), orient="index")
    write_to_file(create_kg_data(df_train), "drugprot_kg_train.tsv")
    df_dev = pd.DataFrame.from_dict(load_from_bin("dev_org"), orient="index")
    write_to_file(create_kg_data(df_dev), "drugprot_kg_dev.tsv")


if __name__ == '__main__':
    main()
