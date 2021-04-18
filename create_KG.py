from data_inspection import get_df_from_data
import networkx as nx
import matplotlib.pyplot as plt

abs_file_name, abs_df, ent_file_name, ent_df, gs_file_name, gs_df, rel_file_name, rel_df, pred_file_name, pred_df = \
    get_df_from_data("sample")

kg_df = rel_df.copy()
print(kg_df)
kg_df["Arg1"] = kg_df["Arg1"].apply(lambda x: x[5:])
kg_df["Arg2"] = kg_df["Arg2"].apply(lambda x: x[5:])
# TODO: drop N entities
index_names = kg_df[kg_df["Evaluation Type"] == "N"].index
kg_df.drop(index_names, inplace=True)
print(kg_df)

G = nx.from_pandas_edgelist(kg_df, source="Arg1", target="Arg2", edge_attr="CPR Group", create_using=nx.MultiDiGraph)
print(G.number_of_nodes())
print(G.number_of_edges())
# print(list(G.nodes))
# print(list(G.edges))

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5)  # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos=pos)
plt.show()
