from drugprot_preprocess import get_df_from_data_test_large, create_data_dict, prepare_data
import pandas as pd
import logging
from util import load_from_bin
from drugprot_lm_predict import lm_predict
from pykeen_embd import prepare_embd
from tfdf_predict import tfdf_predict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def process_test_set_large(lm_model_name, annotation, kg_model_name=None, on_tiny=False):
    args = {
        "model_name": lm_model_name,
        "annotation": annotation,
        "dataset": "test_large",
        "max_length": 192,
        "on_tiny": on_tiny,
        "batch_size": 8,
        "device_ids": [0, 1],
        "local_files_only": True,
    }
    lm_predict(args)
    if kg_model_name:
        prepare_embd(kg_model_name, "test", kg_model_name, annotation, on_tiny=on_tiny)
        tfdf_predict(kg_model_name, "test", lm_model_name, annotation, on_tiny=on_tiny)


def export_data_large(lm_model_name, annotation, file_name, kg_model_name=None, on_tiny=False):
    model_ann_name = lm_model_name + "_" + annotation + "_tiny_ft" \
        if on_tiny else lm_model_name + "_" + annotation + "_ft"
    label_name = f"relation_tfdf_{kg_model_name}_{model_ann_name}" \
        if kg_model_name else f"relation_{model_ann_name}"
    dataset = "test_large_tiny" if on_tiny else "test_large"
    df = load_from_bin(dataset)
    df = df[["pmid", label_name, "Arg1", "Arg2"]]
    # drop none relation
    logging.info(f"shape before dropping none relations: {df.shape}")
    df = df.drop(df[df[label_name] == "NONE"].index)
    logging.info(f"shape after dropping none relations: {df.shape}")
    path = f"data/drugprot_preprocessed/{file_name}"
    df.to_csv(path, sep="\t", header=False, index=False)
    logging.info(f"file exported in {path}")


def main():
    # preprocess data for test set
    dataset = "test_large"
    abs_df, ent_df = get_df_from_data_test_large()
    # test set doesn't have relation tsv file, using place holder dataframe
    column_names = ["a", "b", "c"]
    rel_df = pd.DataFrame(columns=column_names)
    data_dict = create_data_dict(abs_df, ent_df, rel_df)
    prepare_data(data_dict, dataset=dataset)

    lm_model_name = "allenai/scibert_scivocab_uncased"
    annotation = "sci"
    kg_model_name = None
    on_tiny = False
    file_name = "test_large_predict_scibert_sci_v2.0.tsv"
    process_test_set_large(lm_model_name, annotation, kg_model_name, on_tiny)
    export_data_large(lm_model_name, annotation, file_name, kg_model_name, on_tiny)

    lm_model_name = "allenai/scibert_scivocab_uncased"
    annotation = "sci"
    kg_model_name = "TransE"
    on_tiny = False
    file_name = "test_large_predict_scibert_sci_transe_v2.0.tsv"
    process_test_set_large(lm_model_name, annotation, kg_model_name, on_tiny)
    export_data_large(lm_model_name, annotation, file_name, kg_model_name, on_tiny)


if __name__ == '__main__':
    main()
