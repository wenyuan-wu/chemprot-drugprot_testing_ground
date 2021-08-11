from util import get_df_from_data
from spacy.lang.en import English

data_set = "training"
abs_df, ent_df, rel_df = get_df_from_data(data_set)
# abs_df: [3500 rows x 3 columns]
# ent_df: [89529 rows x 6 columns]
# rel_df: [17274 rows x 4 columns]

pmids = abs_df.index.values.tolist()  # len(pmids): 3500

pmid = 23017395
complete = " ".join([abs_df.at[pmid, "Title"], abs_df.at[pmid, "Abstract"]])

offset_to_ent_dict = {}
for index, row in ent_df.loc[pmid].iterrows():
    key = range(row["Start"], row["End"])
    offset_to_ent_dict[key] = row.to_dict()

rel_dict = {}
for index, row in rel_df.loc[pmid].iterrows():
    key = (row["Arg1"][5:], row["Arg2"][5:])
    rel_dict[key] = row.to_dict()


def check_sub_range(range_1, range_2):
    return set(range_1).intersection(set(range_2))


nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp(complete)

soi = 0
data_dict = {}
# TODO: debug
for idx, sent in enumerate(doc.sents):
    sent = sent.text + " "
    eoi = len(sent) + soi
    sent_range = range(soi, eoi)
    soi = eoi

    # check entities
    for key, val in offset_to_ent_dict.items():
        ent_count = 0
        ent_dict = {}
        if check_sub_range(sent_range, key):
            ent_dict = {val["Entity #"]: val}
            ent_count += 1

            # some annotation could happen here
            # check relations
            rel_count = 0

            for k, v in rel_dict.items():
                if k[0] in ent_dict.keys() and k[1] in ent_dict.keys():
                    rel_count += 1
                    sent_id = int(str(pmid) + str(idx) + str(rel_count))
                    data_dict[sent_id] = {"text_raw": sent,
                                          "ent_count": ent_count,
                                          "ent_dict": ent_dict,
                                          "Relation": {v["Relation"]: v}
                                          }
                else:
                    sent_id = int(str(pmid) + str(idx) + str(rel_count))
                    data_dict[sent_id] = {"text_raw": sent,
                                          "ent_count": ent_count,
                                          "ent_dict": ent_dict,
                                          "Relation": {}
                                          }
    # different kinds of annotations here

print(data_dict)
