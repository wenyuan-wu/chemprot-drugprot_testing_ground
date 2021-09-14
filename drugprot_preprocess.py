from spacy.lang.en import English
from os.path import join
import logging
from tqdm import tqdm
import pandas as pd
from util import save_to_bin
from pprint import pprint

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def get_df_from_data(data_set="training"):
    """
    Function to create pandas DataFrame from data files
    :param data_set: string, type for dataset ["training", "development"]
    :return: pandas dataframe w.r.t. dataset
    """
    data_path = join("data", "drugprot", data_set)
    # abstracts
    abs_file_name = join(data_path, "drugprot_" + data_set + "_abstracts" + ".tsv")
    logging.info(f"loading data from {abs_file_name}")
    abs_col_names = ["Title", "Abstract"]
    abs_df = pd.read_csv(abs_file_name, sep="\t", names=abs_col_names, index_col=0, keep_default_na=False)

    # entity mention annotations
    ent_file_name = join(data_path, "drugprot_" + data_set + "_entities" + ".tsv")
    logging.info(f"loading data from {ent_file_name}")
    ent_col_names = ["Entity #", "Type", "Start", "End", "Text"]
    ent_df = pd.read_csv(ent_file_name, sep="\t", names=ent_col_names, index_col=0, keep_default_na=False)

    # drugprot detailed relation annotations
    rel_file_name = join(data_path, "drugprot_" + data_set + "_relations" + ".tsv")
    logging.info(f"loading data from {rel_file_name}")
    rel_col_names = ["Relation", "Arg1", "Arg2"]
    rel_df = pd.read_csv(rel_file_name, sep="\t", names=rel_col_names, index_col=0, keep_default_na=False)

    return abs_df, ent_df, rel_df


def get_df_from_data_test():
    """
    Function to create pandas DataFrame from data files, test set
    :return: pandas dataframe w.r.t. dataset
    """
    # Drugprot test set
    data_path = join("data", "drugprot", "test-background")

    # abstracts
    abs_file_name = join(data_path, "test_background" + "_abstracts" + ".tsv")
    logging.info(f"loading data from {abs_file_name}")
    abs_col_names = ["Title", "Abstract"]
    abs_df = pd.read_csv(abs_file_name, sep="\t", names=abs_col_names, index_col=0, keep_default_na=False)

    # entity mention annotations
    ent_file_name = join(data_path, "test_background" + "_entities" + ".tsv")
    logging.info(f"loading data from {ent_file_name}")
    ent_col_names = ["Entity #", "Type", "Start", "End", "Text"]
    ent_df = pd.read_csv(ent_file_name, sep="\t", names=ent_col_names, index_col=0, keep_default_na=False)

    return abs_df, ent_df


def check_sub_range(range_1, range_2):
    """
    Function to check if two range has intersection
    :param range_1: first range
    :param range_2: second range
    :return: boolean
    """
    return set(range_1).intersection(set(range_2))


def remove_tag():
    # TODO: remove tag in test set
    raise NotImplementedError


def sent_annotation(gene, chem, sent, soi, annotation):
    """
    Annotate sentence in different styles, i.e. scibert, biobert, etc.
    :param gene: gene expression
    :param chem: chemical expression
    :param sent: sentence, string
    :param soi: beginning of the sentence indicator
    :param annotation: annotation style
    :return: annotated sentence
    """
    sent_list = []
    # update ent_dict to focus on the only two entities involved in the relation
    ent_dict = {gene["Entity #"]: gene, chem["Entity #"]: chem}
    srt_list = sorted(ent_dict.items(), key=lambda x: x[1]["Start"], reverse=False)

    if annotation == "sci":
        """
        idx_1: start of the first entity
        char_1: annotation for the start character of the first entity, depending on the type
        idx_2: end the first entity
        char_2: annotation for the end character of the first entity, depending on the type
        idx_3, char_3, idx_4, char_4: same deal to the second character
        """
        idx_1 = srt_list[0][1]["Start"] - soi
        char_1 = "<< " if srt_list[0][1]["Type"].startswith("CHEM") else "[[ "
        idx_2 = srt_list[0][1]["End"] - soi - 1
        char_2 = " >>" if srt_list[0][1]["Type"].startswith("CHEM") else " ]]"
        idx_3 = srt_list[1][1]["Start"] - soi
        char_3 = "<< " if srt_list[1][1]["Type"].startswith("CHEM") else "[[ "
        idx_4 = srt_list[1][1]["End"] - soi - 1
        char_4 = " >>" if srt_list[1][1]["Type"].startswith("CHEM") else " ]]"
        for idx, char in enumerate(list(sent)):
            if idx == idx_1:
                char = char_1 + char
            elif idx == idx_2:
                char = char + char_2
            elif idx == idx_3:
                char = char_3 + char
            elif idx == idx_4:
                char = char + char_4
            sent_list.append(char)

    elif annotation == "bio":
        """
        idx_1: start of the first entity
        idx_2: end the first entity
        idx_range_1: index span of the first entity
        char_1: annotation of the first entity, depending on the type
        idx_3, idx_4, idx_range_2, char_2: same deal to the second character
        """
        idx_1 = srt_list[0][1]["Start"] - soi
        idx_2 = srt_list[0][1]["End"] - soi - 1
        char_1 = "$CHEMICAL#" if srt_list[0][1]["Type"].startswith("CHEM") else "$GENE#"
        idx_range_1 = list(range(idx_1 + 1, idx_2 + 1))
        idx_3 = srt_list[1][1]["Start"] - soi
        idx_4 = srt_list[1][1]["End"] - soi - 1
        idx_range_2 = list(range(idx_3 + 1, idx_4 + 1))
        char_2 = "$CHEMICAL#" if srt_list[1][1]["Type"].startswith("CHEM") else "$GENE#"
        for idx, char in enumerate(list(sent)):
            if idx == idx_1:
                char = char_1
            elif idx in idx_range_1:
                char = ""
            elif idx == idx_3:
                char = char_2
            elif idx in idx_range_2:
                char = ""
            sent_list.append(char)

    return "".join(sent_list)


def create_data_dict(abs_df, ent_df, rel_df):
    """
    Create training data from raw files with different annotation styles
    :param abs_df: abstract dataframe
    :param ent_df: entities dataframe
    :param rel_df: relations dataframe
    :return: dictionary of preprocessed data
    """
    # use spacy to split abstracts into sentences
    nlp = English()
    nlp.add_pipe("sentencizer")
    pmids = abs_df.index.values.tolist()  # training set pmids: 3500
    logging.info(f"Number of PMIDs: {len(pmids)}")
    # getting none distribution information
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
            ent_dict = {}
            gene_list = []
            chem_list = []
            for key, val in offset_to_ent_dict.items():
                if check_sub_range(sent_range, key) and val["Type"].startswith("GENE"):
                    gene_list.append(val)
                    ent_dict[val["Entity #"]] = val
                elif check_sub_range(sent_range, key) and val["Type"].startswith("CHEM"):
                    chem_list.append(val)
                    ent_dict[val["Entity #"]] = val

            if gene_list and chem_list:
                itr_count = 0
                for gene in gene_list:
                    for chem in chem_list:
                        rel_tup = (chem['Entity #'], gene['Entity #'])
                        sent_id = int(str(pmid) + str(idx) + str(itr_count))
                        itr_count += 1
                        # annotation
                        sent_sci = sent_annotation(gene, chem, sent, soi, annotation="sci")
                        sent_bio = sent_annotation(gene, chem, sent, soi, annotation="bio")
                        data_dict[sent_id] = {"pmid": pmid,
                                              "ent_dict": {gene["Entity #"]: gene, chem["Entity #"]: chem},
                                              "itr_count": itr_count,
                                              "text_raw": sent,
                                              "text_sci": sent_sci,
                                              "text_bio": sent_bio,
                                              "Ent1": chem["Text"],
                                              "Ent2": gene["Text"]
                                              }
                        # check if relation exists in this sentence
                        if rel_tup in rel_dict.keys():
                            pos_count += 1
                            data_dict[sent_id]["Arg1"] = rel_dict[rel_tup]["Arg1"]
                            data_dict[sent_id]["Arg2"] = rel_dict[rel_tup]["Arg2"]
                            data_dict[sent_id]["relation"] = rel_dict[rel_tup]["Relation"]
                        else:
                            neg_count += 1
                            data_dict[sent_id]["Arg1"] = f"Arg1:{chem['Entity #']}"
                            data_dict[sent_id]["Arg2"] = f"Arg2:{gene['Entity #']}"
                            data_dict[sent_id]["relation"] = "NONE"

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

    data_set = "development"
    abs_df, ent_df, rel_df = get_df_from_data(data_set)
    data_dict_dev = create_data_dict(abs_df, ent_df, rel_df)
    save_to_bin(data_dict_dev, "dev_org")
    # for debug purpose
    # pprint(data_dict_dev)

    # preprocess data for test set
    abs_df, ent_df = get_df_from_data_test()
    column_names = ["a", "b", "c"]
    rel_df = pd.DataFrame(columns=column_names)
    data_dict_test = create_data_dict(abs_df, ent_df, rel_df)
    save_to_bin(data_dict_test, "test_org")


if __name__ == '__main__':
    main()
