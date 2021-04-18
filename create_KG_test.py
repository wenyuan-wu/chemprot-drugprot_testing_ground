import pandas as pd
import networkx as nx

col = []
gs_df = pd.read_csv('data/chemprot_sample/chemprot_sample_gold_standard.tsv', sep='\t', header=None)
print()

G = nx.Graph()
G.add_node(1)

