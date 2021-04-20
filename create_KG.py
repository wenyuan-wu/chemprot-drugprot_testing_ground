from data_inspection import get_df_from_data
import networkx as nx
import matplotlib.pyplot as plt

abs_file_name, abs_df, ent_file_name, ent_df, gs_file_name, gs_df, rel_file_name, rel_df, pred_file_name, pred_df = \
    get_df_from_data("sample")

# create dataframe for knowledge graph
kg_df = gs_df.copy()
kg_df["Arg1"] = kg_df["Arg1"].apply(lambda x: x[5:])
kg_df["Arg2"] = kg_df["Arg2"].apply(lambda x: x[5:])

# drop entities marked with "N" in evaluation type, note that there's a space after "N" in original data
# index_names = kg_df[kg_df["Evaluation Type"] == "N "].index
# kg_df.drop(index_names, inplace=True)

pmids = kg_df["PMID"].unique()

# pmid = 23538162
# ent_nr = "T1"
# test_str = ent_df.loc[(ent_df["PMID"] == pmid) & (ent_df["Entity #"] == ent_nr), "Text"].values
# print(test_str)

for pmid in pmids:
    ents_1 = kg_df.loc[(kg_df["PMID"] == pmid), "Arg1"].values
    ents_2 = kg_df.loc[(kg_df["PMID"] == pmid), "Arg2"].values
    for ent_nr in ents_1:
        ent_str = ent_df.loc[(ent_df["PMID"] == pmid) & (ent_df["Entity #"] == ent_nr), "Text"].values
        kg_df.loc[(kg_df["PMID"] == pmid) & (kg_df["Arg1"] == ent_nr), "Arg1"] = ent_str[0]
    for ent_nr in ents_2:
        ent_str = ent_df.loc[(ent_df["PMID"] == pmid) & (ent_df["Entity #"] == ent_nr), "Text"].values
        kg_df.loc[(kg_df["PMID"] == pmid) & (kg_df["Arg2"] == ent_nr), "Arg2"] = ent_str[0]
print(kg_df)

# create knowledge graph from dataframe
G = nx.from_pandas_edgelist(kg_df, source="Arg1", target="Arg2", edge_attr="CPR Group", create_using=nx.MultiDiGraph)
# print(G.number_of_nodes())
# print(G.number_of_edges())
# print(list(G.nodes))
# print(list(G.edges))

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5)  # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)
plt.savefig("KG.png")
