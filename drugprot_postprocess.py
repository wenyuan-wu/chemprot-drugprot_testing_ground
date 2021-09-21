from util import load_from_bin
from pprint import pprint


def export_data():
    pass




df = load_from_bin("test_sci_lm_kg_dp")
print(df.columns)
df = df[["pmid", "relation_pred", "Arg1", "Arg2"]]
# drop none relation
print(df.shape)
df = df.drop(df[df["relation_pred"] == "NONE"].index)
print("after dropping:")
print(df.shape)

print(df.head(3))

path = "data/drugprot_preprocessed/test_pred.tsv"
df.to_csv(path, sep="\t", header=False, index=False)
