from spacy.lang.en import English
from os.path import join
import logging
from tqdm import tqdm
import pandas as pd
from util import save_to_bin
import pprint

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def get_df_from_data(data_set="training"):
    """
    Function to create pandas DataFrame from data files
    :param data_set: string, type for dataset ["training", "development", "test"]
    :return: pandas dataframe w.r.t. dataset
    """
    data_path = join("data", "drugprot", data_set)
    # TODO: add logging info and doc string
    # abstracts
    abs_file_name = join(data_path, "drugprot_" + data_set + "_abstracts" + ".tsv")
    logging.info(f"loading data from {abs_file_name}")
    abs_col_names = ["Title", "Abstract"]
    abs_df = pd.read_csv(abs_file_name, sep="\t", names=abs_col_names, index_col=0)

    # entity mention annotations
    ent_file_name = join(data_path, "drugprot_" + data_set + "_entities" + ".tsv")
    ent_col_names = ["Entity #", "Type", "Start", "End", "Text"]
    ent_df = pd.read_csv(ent_file_name, sep="\t", names=ent_col_names, index_col=0)

    # drugprot detailed relation annotations
    rel_file_name = join(data_path, "drugprot_" + data_set + "_relations" + ".tsv")
    rel_col_names = ["Relation", "Arg1", "Arg2"]
    rel_df = pd.read_csv(rel_file_name, sep="\t", names=rel_col_names, index_col=0)

    return abs_df, ent_df, rel_df


def get_df_from_data_test():
    """
    Function to create pandas DataFrame from data files
    :param data_set: string, type for dataset ["training", "development", "test"]
    :return: pandas dataframe w.r.t. dataset
    """
    # Drugprot test set
    data_path = join("data", "drugprot", "test-background")

    # abstracts
    abs_file_name = join(data_path, "test_background" + "_abstracts" + ".tsv")
    abs_col_names = ["Title", "Abstract"]
    abs_df = pd.read_csv(abs_file_name, sep="\t", names=abs_col_names, index_col=0)

    # entity mention annotations
    ent_file_name = join(data_path, "test_background" + "_entities" + ".tsv")
    ent_col_names = ["Entity #", "Type", "Start", "End", "Text"]
    ent_df = pd.read_csv(ent_file_name, sep="\t", names=ent_col_names, index_col=0)

    return abs_df, ent_df


def check_sub_range(range_1, range_2):
    return set(range_1).intersection(set(range_2))


def remove_tag():
    # TODO: remove tag in test set
    raise NotImplementedError


def create_data_dict(abs_df, ent_df, rel_df):
    nlp = English()
    nlp.add_pipe("sentencizer")
    pmids = abs_df.index.values.tolist()  # training set pmids: 3500
    logging.info(f"Number of PMIDs: {len(pmids)}")

    data_dict = {}
    for pmid in tqdm(pmids):
        pmid = 17380207
        complete = " ".join([abs_df.at[pmid, "Title"], abs_df.at[pmid, "Abstract"]])

        offset_to_ent_dict = {}
        try:
            for index, row in ent_df.loc[pmid].iterrows():
                key = range(row["Start"], row["End"])
                offset_to_ent_dict[key] = row.to_dict()
        except KeyError:
            offset_to_ent_dict = {range(10240, 10241): {'Entity #': 'Null',
                                                        'Type': 'CHEMICAL',
                                                        'Start': 10240,
                                                        'End': 10241,
                                                        'Text': 'placeholder'}
                                  }
        except AttributeError:
            key = range(ent_df.loc[pmid]["Start"], ent_df.loc[pmid]["End"])
            offset_to_ent_dict[key] = ent_df.loc[pmid].to_dict()

        rel_dict = {}
        try:
            for index, row in rel_df.loc[pmid].iterrows():
                key = (row["Arg1"][5:], row["Arg2"][5:])
                rel_dict[key] = row.to_dict()
        except KeyError:
            rel_dict = {('Null1', 'Null2'): {'Relation': 'Placeholder', 'Arg1': 'Arg1:Null1', 'Arg2': 'Arg2:Null2'}}
        except AttributeError:
            key = (rel_df.loc[pmid]["Arg1"][5:], rel_df.loc[pmid]["Arg2"][5:])
            rel_dict[key] = rel_df.loc[pmid].to_dict()

        doc = nlp(complete)
        soi = 0
        for idx, sent in enumerate(doc.sents):
            sent = sent.text + " "
            eoi = len(sent) + soi
            sent_range = range(soi, eoi)
            soi = eoi

            # check entities
            ent_count = 0
            ent_dict = {}
            for key, val in offset_to_ent_dict.items():
                if check_sub_range(sent_range, key):
                    ent_dict[val["Entity #"]] = val
                    ent_count += 1

            # check relations
            rel_count = 0
            for k, v in rel_dict.items():
                if k[0] in ent_dict.keys() and k[1] in ent_dict.keys():
                    sent_id = int(str(pmid) + str(idx) + str(rel_count))
                    data_dict[sent_id] = {"text_raw": sent,
                                          "ent_count": ent_count,
                                          "ent_dict": ent_dict,
                                          "rel_count": rel_count,
                                          "rel_dict": v,
                                          "relation": v["Relation"],
                                          "pmid": pmid,
                                          }
                    rel_count += 1
                else:
                    sent_id = int(str(pmid) + str(idx) + str(rel_count))
                    data_dict[sent_id] = {"text_raw": sent,
                                          "ent_count": ent_count,
                                          "ent_dict": ent_dict,
                                          "rel_count": rel_count,
                                          "rel_dict": {},
                                          "relation": "NONE",
                                          "pmid": pmid,
                                          }
            # TODO:
            # TODO: some annotation could happen here

        break

    for k, v in data_dict.items():
        print(k, v)

    return data_dict


def main():
    # data_set = "training"
    # abs_df, ent_df, rel_df = get_df_from_data(data_set)
    # data_dict_train = create_data_dict(abs_df, ent_df, rel_df)
    # # pprint.pprint(data_dict_train)
    # save_to_bin(data_dict_train, "train_org")
    # # for debug purpose
    # # pprint.pprint(data_dict_train)

    data_set = "development"
    abs_df, ent_df, rel_df = get_df_from_data(data_set)
    data_dict_dev = create_data_dict(abs_df, ent_df, rel_df)
    # save_to_bin(data_dict_dev, "dev_org")

    # abs_df, ent_df = get_df_from_data_test()
    # column_names = ["a", "b", "c"]
    # rel_df = pd.DataFrame(columns=column_names)
    # data_dict_test = create_data_dict(abs_df, ent_df, rel_df)
    # save_to_bin(data_dict_test, "test_org")


if __name__ == '__main__':
    main()
