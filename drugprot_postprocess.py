import logging
from util import load_from_bin

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def export_data(lm_model_name, annotation, kg_model_name="None", on_tiny=False, on_lm=True):
    model_ann_name = lm_model_name + "_" + annotation + "_tiny_ft" \
        if on_tiny else lm_model_name + "_" + annotation + "_ft"
    label_tfdf_name = f"relation_tfdf_{kg_model_name}_{model_ann_name}"
    label_lm_name = f"relation_{model_ann_name}"
    dataset = "development_tiny" if on_tiny else "development"
    df = load_from_bin(dataset)
    y_pred = label_lm_name if on_lm else label_tfdf_name
    df = df[["pmid", y_pred, "Arg1", "Arg2"]]
    # drop none relation
    logging.info(f"shape before dropping none relations: {df.shape}")
    df = df.drop(df[df[y_pred] == "NONE"].index)
    logging.info(f"shape after dropping none relations: {df.shape}")
    path = "data/drugprot_preprocessed/test_pred.tsv"
    df.to_csv(path, sep="\t", header=False, index=False)
    logging.info(f"file exported in {path}")


def main():
    lm_model_name = ""
    annotation = ""
    kg_model_name = ""
    on_tiny = False
    on_lm = False
    export_data(lm_model_name, annotation, kg_model_name, on_tiny, on_lm)


if __name__ == '__main__':
    main()
