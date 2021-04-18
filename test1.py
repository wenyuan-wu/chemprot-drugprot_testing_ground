from os import listdir
from os.path import join
import pandas as pd

# get information for all data
data_path = "data"
for i in listdir(data_path):
    print(f"Files in folder {i}:")
    for j in listdir(join(data_path, i)):
        if j.endswith(".tsv"):
            file_name = join(data_path, i, j)
            df = pd.read_csv(file_name, sep='\t', header=None)
            print(file_name)
            print(df)
            print("\n")
    print("=" * 20)

