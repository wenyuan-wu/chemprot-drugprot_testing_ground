from os.path import join
import pandas as pd


def get_df_from_data(data_set="training"):
    data_path = join("data", "chemprot_" + data_set)

    # abstracts
    abs_file_name = join(data_path, "chemprot_" + data_set + "_abstracts" + ".tsv")
    abs_col_names = ["PMID", "Title", "Abstract"]
    abs_df = pd.read_csv(abs_file_name, sep="\t", names=abs_col_names)

    # entity mention annotations
    ent_file_name = join(data_path, "chemprot_" + data_set + "_entities" + ".tsv")
    ent_col_names = ["PMID", "Entity #", "Type", "Start", "End", "Text"]
    ent_df = pd.read_csv(ent_file_name, sep="\t", names=ent_col_names)

    # Gold Standard data
    gs_file_name = join(data_path, "chemprot_" + data_set + "_gold_standard" + ".tsv")
    gs_col_names = ["PMID", "CPR Group", "Arg1", "Arg2"]
    gs_df = pd.read_csv(gs_file_name, sep="\t", names=gs_col_names)

    # ChemProt detailed relation annotations
    rel_file_name = join(data_path, "chemprot_" + data_set + "_relations" + ".tsv")
    rel_col_names = ["PMID", "CPR Group", "Evaluation Type", "CPR", "Arg1", "Arg2"]
    rel_df = pd.read_csv(rel_file_name, sep="\t", names=rel_col_names)

    if data_set == "sample":
        # ChemProt task predictions
        pred_file_name = join(data_path, "chemprot_" + data_set + "_predictions" + ".tsv")
        pred_col_names = ["PMID", "CPR Group", "Arg1", "Arg2"]
        pred_df = pd.read_csv(pred_file_name, sep="\t", names=pred_col_names)
        return abs_file_name, abs_df, ent_file_name, ent_df, gs_file_name, gs_df, rel_file_name, rel_df, \
               pred_file_name, pred_df

    return abs_file_name, abs_df, ent_file_name, ent_df, gs_file_name, gs_df, rel_file_name, rel_df


def get_df_from_data_test(data_set="test"):
    data_path = join("data", "chemprot_" + data_set + "_gs")

    # abstracts
    abs_file_name = join(data_path, "chemprot_" + data_set + "_abstracts_gs" + ".tsv")
    abs_col_names = ["PMID", "Title", "Abstract"]
    abs_df = pd.read_csv(abs_file_name, sep="\t", names=abs_col_names)

    # entity mention annotations
    ent_file_name = join(data_path, "chemprot_" + data_set + "_entities_gs" + ".tsv")
    ent_col_names = ["PMID", "Entity #", "Type", "Start", "End", "Text"]
    ent_df = pd.read_csv(ent_file_name, sep="\t", names=ent_col_names)

    # Gold Standard data
    gs_file_name = join(data_path, "chemprot_" + data_set + "_gold_standard" + ".tsv")
    gs_col_names = ["PMID", "CPR Group", "Arg1", "Arg2"]
    gs_df = pd.read_csv(gs_file_name, sep="\t", names=gs_col_names)

    # ChemProt detailed relation annotations
    rel_file_name = join(data_path, "chemprot_" + data_set + "_relations_gs" + ".tsv")
    rel_col_names = ["PMID", "CPR Group", "Evaluation Type", "CPR", "Arg1", "Arg2"]
    rel_df = pd.read_csv(rel_file_name, sep="\t", names=rel_col_names)

    return abs_file_name, abs_df, ent_file_name, ent_df, gs_file_name, gs_df, rel_file_name, rel_df


def main():
    data_set = "sample"
    # data_set = "training"
    # data_set = "development"
    for i in get_df_from_data(data_set):
        print(i)
        print()

    for i in get_df_from_data_test("test"):
        print(i)
        print()


if __name__ == "__main__":
    main()
