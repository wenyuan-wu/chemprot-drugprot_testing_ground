import numpy as np
import logging
from util import load_from_bin, save_to_bin
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def label_evaluate(lm_model_name, annotation, kg_model_name="None", on_tiny=False, on_lm=True):
    model_ann_name = lm_model_name + "_" + annotation + "_tiny_ft" \
        if on_tiny else lm_model_name + "_" + annotation + "_ft"
    label_tfdf_name = f"label_tfdf_{kg_model_name}_{model_ann_name}"
    label_lm_name = f"label_{model_ann_name}"
    dataset = "development_tiny" if on_tiny else "development"
    df = load_from_bin(dataset)
    y_true = df["label"]
    y_pred = df[label_lm_name] if on_lm else df[label_tfdf_name]
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average="micro"), \
           recall_score(y_true, y_pred, average="micro"), f1_score(y_true, y_pred, average="micro")


def main():
    kg_models = ["TransE", "PairRE"]
    models = [
        "bert-base-uncased",
        "allenai/scibert_scivocab_uncased",
        "dmis-lab/biobert-base-cased-v1.1",
    ]
    annotations = ["raw", "sci", "bio"]

    for lm_model in models:
        for annotation in annotations:
            logging.info(f"model: {lm_model}, annotation: {annotation}")
            # result = label_evaluate(lm_model, annotation, on_tiny=True, on_lm=True)
            result = label_evaluate(lm_model, annotation, on_tiny=False, on_lm=True)
            logging.info("accuracy: {:.3f}".format(result[0]))
            logging.info("precision: {:.3f}".format(result[1]))
            logging.info("recall: {:.3f}".format(result[2]))
            logging.info("f1 score: {:.3f}".format(result[3]))
    #
    # for kg_model in kg_models:
    #     for lm_model in models:
    #         for annotation in annotations:
    #             logging.info(f"kg_model: {kg_model}, lm_model: {lm_model}, annotation: {annotation}")
    #             result = label_evaluate(lm_model, annotation, kg_model, on_tiny=True, on_lm=False)
    #             # result = label_evaluate(lm_model, annotation, kg_model, on_tiny=False, on_lm=False)
    #             logging.info("accuracy: {:.3f}".format(result[0]))
    #             logging.info("precision: {:.3f}".format(result[1]))
    #             logging.info("recall: {:.3f}".format(result[2]))
    #             logging.info("f1 score: {:.3f}".format(result[3]))


if __name__ == '__main__':
    main()
