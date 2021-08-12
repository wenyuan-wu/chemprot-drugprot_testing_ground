from util import get_df_from_data
from spacy.lang.en import English
import pprint
import logging
from tqdm import tqdm
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    datefmt='%d-%b-%y %H:%M:%S'
                    )


def check_sub_range(range_1, range_2):
    return set(range_1).intersection(set(range_2))


def create_data_dict(abs_df, ent_df, rel_df, annotation="None"):
    nlp = English()
    nlp.add_pipe("sentencizer")
    pmids = abs_df.index.values.tolist()  # training set pmids: 3500
    logging.info(f"Number of PMIDs: {len(pmids)}")

    data_dict = {}
    for pmid in tqdm(pmids):
        complete = " ".join([abs_df.at[pmid, "Title"], abs_df.at[pmid, "Abstract"]])

        offset_to_ent_dict = {}
        try:
            for index, row in ent_df.loc[pmid].iterrows():
                key = range(row["Start"], row["End"])
                offset_to_ent_dict[key] = row.to_dict()
        except KeyError:
            offset_to_ent_dict = {}
        except AttributeError:
            key = range(ent_df.loc[pmid]["Start"], ent_df.loc[pmid]["End"])
            offset_to_ent_dict[key] = ent_df.loc[pmid].to_dict()

        rel_dict = {}
        try:
            for index, row in rel_df.loc[pmid].iterrows():
                key = (row["Arg1"][5:], row["Arg2"][5:])
                rel_dict[key] = row.to_dict()
        except KeyError:
            rel_dict = {}
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
                                          "Relation": {v["Relation"]: v}
                                          }
                    rel_count += 1
                else:
                    sent_id = int(str(pmid) + str(idx) + str(rel_count))
                    data_dict[sent_id] = {"text_raw": sent,
                                          "ent_count": ent_count,
                                          "ent_dict": ent_dict,
                                          "rel_count": rel_count,
                                          "Relation": {}
                                          }

    # some annotation could happen here
    if annotation == "anoy":
        pass

    return data_dict


def main():
    data_set = "training"
    abs_df, ent_df, rel_df = get_df_from_data(data_set)
    # abs_df: [3500 rows x 3 columns]
    # ent_df: [89529 rows x 6 columns]
    # rel_df: [17274 rows x 4 columns]
    data_dict_train = create_data_dict(abs_df, ent_df, rel_df)
    pprint.pprint(data_dict_train)

    data_set = "development"
    abs_df, ent_df, rel_df = get_df_from_data(data_set)
    # abs_df: [3500 rows x 3 columns]
    # ent_df: [89529 rows x 6 columns]
    # rel_df: [17274 rows x 4 columns]
    data_dict_dev = create_data_dict(abs_df, ent_df, rel_df)
    pprint.pprint(data_dict_dev)

    # TODO: get test set right
    data_set = "test"
    abs_df, ent_df = get_df_from_data(data_set)
    column_names = ["a", "b", "c"]
    rel_df = pd.DataFrame(columns = column_names)
    # abs_df: [3500 rows x 3 columns]
    # ent_df: [89529 rows x 6 columns]
    # rel_df: [17274 rows x 4 columns]
    data_dict_test = create_data_dict(abs_df, ent_df, rel_df)
    pprint.pprint(data_dict_test)


if __name__ == '__main__':
    main()

