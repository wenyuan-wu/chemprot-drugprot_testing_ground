import tensorflow_decision_forests as tfdf
import logging
import pandas as pd
import numpy as np
from util import load_from_bin, save_to_bin
import tensorflow as tf

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def tfdf_train(kg_model, dataset, lm_model, annotation, on_tiny=False):
    logging.info(f'Found TF-DF v{tfdf.__version__}')
    dataset = dataset + "_tiny" if on_tiny else dataset
    model_ann_name = lm_model + "_" + annotation + "_tiny_ft" \
        if on_tiny else lm_model + "_" + annotation + "_ft"
    df = load_from_bin(dataset)
    embd_name = f"combi_embd_{kg_model}_{model_ann_name}"
    label_name = f"label_{model_ann_name}"
    x = np.asarray(list(df[embd_name].values)).astype(np.float32)
    y = np.asarray(list(df[label_name].values)).astype(np.int64)
    # Specify the model.
    model = tfdf.keras.RandomForestModel()
    # Optionally, add evaluation metrics.
    model.compile(metrics=["accuracy"])
    model.fit(x, y)
    model.save(f"model/tfdf_{kg_model}_{model_ann_name}")
    return model


def tfdf_predict(model, kg_model, dataset, lm_model, annotation, on_tiny=False):
    dataset = dataset + "_tiny" if on_tiny else dataset
    model_ann_name = lm_model + "_" + annotation + "_tiny_ft" \
        if on_tiny else lm_model + "_" + annotation + "_ft"
    df = load_from_bin(dataset)
    embd_name = f"combi_embd_{kg_model}_{model_ann_name}"
    x_test = np.asarray(list(df[embd_name].values)).astype(np.float32)
    pred = model.predict(x_test)
    predicted_labels = np.argmax(pred, axis=1)
    logging.info(f"predicted labels shape: {predicted_labels.shape}")
    idx_to_label_dict = load_from_bin("idx_to_label_dict")
    df[f"label_tfdf_{kg_model}_{model_ann_name}"] = predicted_labels
    df[f"relation_tfdf_{kg_model}_{model_ann_name}"] = \
        df[f"label_tfdf_{kg_model}_{model_ann_name}"].map(idx_to_label_dict)
    save_to_bin(df, dataset)


def main():
    kg_models = ["TransE", "PairRE"]
    models = [
        "bert-base-uncased",
        "allenai/scibert_scivocab_uncased",
        "dmis-lab/biobert-base-cased-v1.1",
    ]
    annotations = ["raw", "sci", "bio"]
    datasets = ["development", "test"]
    for kg_model in kg_models:
        for lm_model in models:
            for annotation in annotations:
                # model = tfdf_train(kg_model, "training", lm_model, annotation, True)
                model = tfdf_train(kg_model, "training", lm_model, annotation)
                for dataset in datasets:
                    logging.info(f"kg_model: {kg_model}, dataset: {dataset}")
                    logging.info(f"lm_model: {lm_model}, annotation: {annotation}")
                    # tfdf_predict(model, kg_model, dataset, lm_model, annotation, True)
                    tfdf_predict(model, kg_model, dataset, lm_model, annotation)


if __name__ == '__main__':
    main()
