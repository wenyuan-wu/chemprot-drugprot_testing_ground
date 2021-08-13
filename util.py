from os.path import join
import pandas as pd


# def get_df_from_data(data_set="training"):
#     """
#     Function to create pandas DataFrame from data files
#     :param data_set: string, type for dataset ["training", "development", "test"]
#     :return: pandas dataframe w.r.t. dataset
#     """
#     data_path = join("data", "drugprot", data_set)
#
#     # abstracts
#     abs_file_name = join(data_path, "drugprot_" + data_set + "_abstracts" + ".tsv")
#     abs_col_names = ["Title", "Abstract"]
#     abs_df = pd.read_csv(abs_file_name, sep="\t", names=abs_col_names, index_col=0)
#
#     # entity mention annotations
#     ent_file_name = join(data_path, "drugprot_" + data_set + "_entities" + ".tsv")
#     ent_col_names = ["Entity #", "Type", "Start", "End", "Text"]
#     ent_df = pd.read_csv(ent_file_name, sep="\t", names=ent_col_names, index_col=0)
#
#     # drugprot detailed relation annotations
#     rel_file_name = join(data_path, "drugprot_" + data_set + "_relations" + ".tsv")
#     rel_col_names = ["Relation", "Arg1", "Arg2"]
#     rel_df = pd.read_csv(rel_file_name, sep="\t", names=rel_col_names, index_col=0)
#
#     return abs_df, ent_df, rel_df
#
#
# def get_df_from_data_test(data_set="test"):
#     """
#     Function to create pandas DataFrame from data files
#     :param data_set: string, type for dataset ["training", "development", "test"]
#     :return: pandas dataframe w.r.t. dataset
#     """
#     # Drugprot test set
#     data_path = join("data", "drugprot", "test-background")
#
#     # abstracts
#     abs_file_name = join(data_path, "test_background" + "_abstracts" + ".tsv")
#     abs_col_names = ["Title", "Abstract"]
#     abs_df = pd.read_csv(abs_file_name, sep="\t", names=abs_col_names, index_col=0)
#
#     # entity mention annotations
#     ent_file_name = join(data_path, "test_background" + "_entities" + ".tsv")
#     ent_col_names = ["Entity #", "Type", "Start", "End", "Text"]
#     ent_df = pd.read_csv(ent_file_name, sep="\t", names=ent_col_names, index_col=0)
#
#     return abs_df, ent_df
