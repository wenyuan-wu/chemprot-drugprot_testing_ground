import logging
import pandas as pd
from util import load_from_bin
from os.path import join

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def create_kg_data(dataframe: pd.DataFrame, contains_none=True):
    kg_list = []
    err_count = 0
    for idx, row in dataframe.iterrows():
        # if row["relation"] != "NONE":
        ent_dict = row['ent_dict']
        for k, v in ent_dict.items():
            if v["Type"].startswith("CHEMICAL"):
                try:
                    arg1 = v["Text"].lower()
                except AttributeError:
                    logging.error(f"unexpected value {v['Text']}")
                    err_count += 1
            else:
                try:
                    arg2 = v["Text"].lower()
                except AttributeError:
                    logging.error(f"unexpected value {v['Text']}")
                    err_count += 1
        kg_list.append((arg1, row["relation"], arg2))
    logging.info(f"Number of unexpected values: {err_count}")
    return kg_list


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
    # print(df_train.ent_dict)
    write_to_file(create_kg_data(df_train), "drugprot_kg_train.tsv")

    df_dev = pd.DataFrame.from_dict(load_from_bin("dev_org"), orient="index")
    write_to_file(create_kg_data(df_dev), "drugprot_kg_dev.tsv")


if __name__ == '__main__':
    main()
