import pandas as pd

ent_col_names = ["Entity #", "Type", "Start", "End", "Text"]
ent_df = pd.read_csv("data/drugprot/training/drugprot_training_entities.tsv",
                     sep="\t",
                     names=ent_col_names,
                     index_col=0,
                     keep_default_na=False)

err = 0
id = 0
for idx, row in ent_df.iterrows():
    if not isinstance(row["Text"], str):
        print("fuck")
        print(row)
        err += 1
    id += 1

print(f"err: {err}, ratio: {err/ (id + 1)}")


