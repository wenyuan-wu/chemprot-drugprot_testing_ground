from spacy.lang.en import English
from os.path import join
import logging
from tqdm import tqdm
import pandas as pd
from typing import Tuple, Any
from util import save_to_bin, load_from_bin
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def get_df_from_data(data_set="training") -> Tuple[Any, Any, Any]:
    """
    Function to create pandas DataFrame from tsv data files
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


def get_df_from_data_test() -> Tuple[Any, Any]:
    """
    Function to create pandas DataFrame from tsv data files, test set only
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


def check_sub_range(range_1: range, range_2: range) -> set:
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


def sent_annotation(gene: dict, chem: dict, sent: str, soi: int, annotation: str) -> str:
    """
    Annotate sentence in different styles, i.e. scibert, biobert, etc.
    :param gene: gene expression, dictionary
    :param chem: chemical expression, dictionary
    :param sent: sentence, string
    :param soi: beginning of the sentence indicator, integer
    :param annotation: annotation style, string
    :return: annotated sentence, string
    """
    # update ent_dict to focus on the only two entities involved in the relation
    ent_dict = {gene["Entity #"]: gene, chem["Entity #"]: chem}
    srt_list = sorted(ent_dict.items(), key=lambda x: x[1]["Start"], reverse=False)
    sent_list = []

    if annotation == "sci":
        """
        SciBert style annotation: surround chemicals with "<< >>" and genes with "[[ ]]"
        idx_1: index for the beginning of the first entity
        char_1: annotation for the beginning character of the first entity, depending on the type
        idx_2: index for the ending of the first entity
        char_2: annotation for the ending character of the first entity, depending on the type
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
        BioBert style annotation: anonymize chemicals into "$CHEMICAL#" and genes into "$GENE#"
        idx_1: index for the beginning of the first entity
        idx_2: index for the ending of the first entity
        idx_range_1: index for the span of the first entity
        char_1: annotation of the first entity, depending on the type
        idx_3, idx_4, idx_range_2, char_2: same deal to the second entity
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


def create_data_dict(abs_df: pd.DataFrame, ent_df: pd.DataFrame, rel_df: pd.DataFrame) -> dict:
    """
    Create training data from raw files with different annotation styles
    :param abs_df: abstract dataframe
    :param ent_df: entities dataframe
    :param rel_df: relations dataframe
    :return: dictionary of preprocessed data
    """
    pmids = list(abs_df.index.values)  # training set pmids: 3500
    logging.info(f"Number of PMIDs: {len(pmids)}")
    # use spacy to split text into sentences
    nlp = English()
    nlp.add_pipe("sentencizer")
    # getting none relation distribution information
    pos_count = 0
    neg_count = 0
    data_dict = {}
    for pmid in tqdm(pmids):
        # for debugging purpose
        # pmid = 17380207
        # create entity offset dictionary, key: range(start, end), value: entity information
        offset_to_ent_dict = {}
        try:
            # if there are multiple entities
            for index, row in ent_df.loc[pmid].iterrows():
                key = range(row["Start"], row["End"])
                offset_to_ent_dict[key] = row.to_dict()
        except KeyError:
            # if there are no entities, create dummy dictionary as placeholder
            offset_to_ent_dict = {range(10240, 10241): {'Entity #': 'Null',
                                                        'Type': 'CHEMICAL',
                                                        'Start': 10240,
                                                        'End': 10241,
                                                        'Text': 'placeholder'}}
        except AttributeError:
            # if there is only one entity
            key = range(ent_df.loc[pmid]["Start"], ent_df.loc[pmid]["End"])
            offset_to_ent_dict[key] = ent_df.loc[pmid].to_dict()

        # create relation dictionary, key: tuple(arg1, arg2), value: relation informaiton
        rel_dict = {}
        try:
            # if there are multiple relations
            for index, row in rel_df.loc[pmid].iterrows():
                key = (row["Arg1"][5:], row["Arg2"][5:])
                rel_dict[key] = row.to_dict()
        except KeyError:
            # if there are no relations, create dummy dictionary as placeholder
            rel_dict = {('Null1', 'Null2'): {'Relation': 'Placeholder', 'Arg1': 'Arg1:Null1', 'Arg2': 'Arg2:Null2'}}
        except AttributeError:
            # if there is only one relation
            key = (rel_df.loc[pmid]["Arg1"][5:], rel_df.loc[pmid]["Arg2"][5:])
            rel_dict[key] = rel_df.loc[pmid].to_dict()

        # complete = title + abstract
        complete = " ".join([abs_df.at[pmid, "Title"], abs_df.at[pmid, "Abstract"]])
        doc = nlp(complete)

        # start of the sentence
        soi = 0
        for idx, sent in enumerate(doc.sents):
            sent = sent.text + " "
            eoi = len(sent) + soi
            sent_range = range(soi, eoi)

            # check if entities exist in this sentence
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


def prepare_data(data_dict: dict, dataset="training", annotation="raw",
                 drop_none=False, frac=0.99, random_state=1024) -> None:
    """
    Prepare data for training from preprocessed data dictionary, especially mapping
    :param data_dict: data dictionary from preprocessed tsv files
    :param dataset: type of the dataset, ["train", "dev", "test"]
    :param annotation: annotation style, ["raw", "sci", "bio"]
    :param drop_none: boolean, if True then drop NONE relation in dataframe
    :param frac: fraction to create tiny dataset for testing
    :param random_state: random state for dropping
    :return: None, prepared files will be saved accordingly
    """
    # data_name = dataset + "_org"
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    # ann_style = "text_" + annotation
    # df = df[[ann_style, "relation", "pmid", "Arg1", "Arg2", "Ent1", "Ent2"]]
    if drop_none:
        # drop sentences with NONE relation
        df = df.drop(df[df["relation"] == "NONE"].index)
    if dataset == "training":
        # convert relation into numerical categories
        df.relation = pd.Categorical(df.relation)
        df["label"] = df.relation.cat.codes
        idx_to_label_dict = dict(enumerate(df['relation'].cat.categories))
        label_to_idx_dict = {v: k for k, v in idx_to_label_dict.items()}
        # create tiny dataset for quick testing purpose
        df_tiny = df.drop(df.sample(frac=frac, random_state=random_state).index)
        # saving files as binary format
        save_to_bin(label_to_idx_dict, f"label_to_idx_dict")
        save_to_bin(idx_to_label_dict, f"idx_to_label_dict")
        save_to_bin(df, f"{dataset}")
        save_to_bin(df_tiny, f"{dataset}_tiny")
    else:
        # mapping relations into numerical categories from loaded mapping dictionary
        label_to_idx_dict = load_from_bin(f"label_to_idx_dict")
        df["label"] = df["relation"].map(label_to_idx_dict)
        df_tiny = df.drop(df.sample(frac=frac, random_state=random_state).index)
        save_to_bin(df, f"{dataset}")
        save_to_bin(df_tiny, f"{dataset}_tiny")


def plot_label_dist(dataset="training", drop_none=False) -> plt:
    """
    Plot the distribution information of the dataset
    :param dataset: training or development set
    :param drop_none: boolean, if True then drop NONE relation in dataframe
    :return: matplotlib.pyplot instance
    """
    df = load_from_bin(dataset)
    file_name = f"info/{dataset}.png"
    if drop_none:
        df = df.drop(df[df["relation"] == "NONE"].index)
        file_name = f"info/{dataset}_drop_none.png"
    logging.info(f"distribution info:\n{df['relation'].value_counts()}")
    # plot data
    sns.set(style='darkgrid')
    # Increase the plot size and font size.
    sns.set(font_scale=2)
    plt.rcParams["figure.figsize"] = (50, 25)
    # Plot the number of tokens of each length.
    sns.countplot(x="relation", data=df)
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        # fontweight='light',
        # fontsize='x-large'
    )
    plt.title('Relation Distribution')
    plt.xlabel('Relation')
    plt.ylabel('# of Training Samples')
    plt.savefig(file_name)
    plt.clf()


def main():
    for dataset in ["training", "development"]:
        abs_df, ent_df, rel_df = get_df_from_data(dataset)
        data_dict = create_data_dict(abs_df, ent_df, rel_df)
        prepare_data(data_dict, dataset=dataset)
        # save plotted images of label distribution information
        plot_label_dist(dataset)
        plot_label_dist(dataset, drop_none=True)

    # preprocess data for test set
    dataset = "test"
    abs_df, ent_df = get_df_from_data_test()
    # test set doesn't have relation tsv file, using place holder dataframe
    column_names = ["a", "b", "c"]
    rel_df = pd.DataFrame(columns=column_names)
    data_dict = create_data_dict(abs_df, ent_df, rel_df)
    prepare_data(data_dict, dataset=dataset)


if __name__ == '__main__':
    main()
