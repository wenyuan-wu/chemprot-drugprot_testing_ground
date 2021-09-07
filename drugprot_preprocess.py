from spacy.lang.en import English
from os.path import join
import logging
from tqdm import tqdm
import pandas as pd
from util import save_to_bin, sent_annotation
from pprint import pprint

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
    pos_count = 0
    neg_count = 0
    data_dict = {}
    for pmid in tqdm(pmids):
        # for debug purpose
        # pmid = 17380207
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

            # check entities
            ent_count = 0
            ent_dict = {}
            gene_list = []
            chem_list = []
            for key, val in offset_to_ent_dict.items():
                if check_sub_range(sent_range, key) and val["Type"].startswith("GENE"):
                    gene_list.append(val)
                    ent_dict[val["Entity #"]] = val
                    ent_count += 1
                elif check_sub_range(sent_range, key) and val["Type"].startswith("CHEM"):
                    chem_list.append(val)
                    ent_dict[val["Entity #"]] = val
                    ent_count += 1

            if gene_list and chem_list:
                itr_count = 0
                for gene in gene_list:
                    for chem in chem_list:
                        rel_tup = (chem['Entity #'], gene['Entity #'])
                        if rel_tup in rel_dict.keys():
                            sent_id = int(str(pmid) + str(idx) + str(itr_count))
                            itr_count += 1
                            pos_count += 1
                            # annotation
                            sent_sci = sent_annotation(gene, chem, sent, soi, annotation="scibert")
                            sent_bio = sent_annotation(gene, chem, sent, soi, annotation="biobert")

                            data_dict[sent_id] = {"text_raw": sent,
                                                  "ent_count": ent_count,
                                                  "ent_dict": {gene["Entity #"]: gene, chem["Entity #"]: chem},
                                                  "itr_count": itr_count,
                                                  "rel_dict": rel_dict[rel_tup],
                                                  "relation": rel_dict[rel_tup]["Relation"],
                                                  "pmid": pmid,
                                                  "text_scibert": sent_sci,
                                                  "text_biobert": sent_bio,

                                                  }

                        else:
                            sent_id = int(str(pmid) + str(idx) + str(itr_count))
                            itr_count += 1
                            neg_count += 1
                            # annotation
                            sent_sci = sent_annotation(gene, chem, sent, soi, annotation="scibert")
                            sent_bio = sent_annotation(gene, chem, sent, soi, annotation="biobert")
                            data_dict[sent_id] = {"text_raw": sent,
                                                  "ent_count": ent_count,
                                                  "ent_dict": {gene["Entity #"]: gene, chem["Entity #"]: chem},
                                                  "itr_count": itr_count,
                                                  "rel_dict": "NONE",
                                                  "relation": "NONE",
                                                  "pmid": pmid,
                                                  "text_scibert": sent_sci,
                                                  "text_biobert": sent_bio
                                                  }
            # update range
            soi = eoi
        # for debug purpose
        # pprint(data_dict)
        # break

    # dist info
    logging.info(f"pos: {pos_count}")
    logging.info(f"neg: {neg_count}")

    return data_dict


def main():
    data_set = "training"
    abs_df, ent_df, rel_df = get_df_from_data(data_set)
    data_dict_train = create_data_dict(abs_df, ent_df, rel_df)
    save_to_bin(data_dict_train, "train_org")
    # for debug purpose
    # pprint(data_dict_train)

    data_set = "development"
    abs_df, ent_df, rel_df = get_df_from_data(data_set)
    data_dict_dev = create_data_dict(abs_df, ent_df, rel_df)
    save_to_bin(data_dict_dev, "dev_org")

    abs_df, ent_df = get_df_from_data_test()
    column_names = ["a", "b", "c"]
    rel_df = pd.DataFrame(columns=column_names)
    data_dict_test = create_data_dict(abs_df, ent_df, rel_df)
    save_to_bin(data_dict_test, "test_org")


if __name__ == '__main__':
    main()
