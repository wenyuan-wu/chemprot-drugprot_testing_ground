import numpy as np
import pykeen.utils
from pykeen.pipeline import pipeline
import pandas as pd
from tqdm import tqdm
from util import load_from_bin, save_to_bin
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def get_kg_embd(entity, embd_tensor, result):
    """
    Extract knowledge graph embedding of entity
    :param entity: string, entity name
    :param embd_tensor: compilation of embedding tensor
    :param result: pykeen pipeline result
    :return: numpy array
    """
    idx = result.training.entity_to_id[entity]
    return embd_tensor[idx]


def concatenate_embd(arr1, arr2, arr3):
    """
    Function to concatenate 3 embeddings (numpy arrays)
    :param arr1: first numpy array
    :param arr2: second numpy array
    :param arr3: third numpy array
    :return: concatenated numpy array
    """
    return np.concatenate((arr1, arr2, arr3), axis=None)


def train_kg_model(model_name="PairRE",
                   train_path="data/drugprot_preprocessed/kg/drugprot_kg_train_combi.tsv",
                   test_path="data/drugprot_preprocessed/kg/drugprot_kg_dev.tsv"):
    """
    Function to train a knowledge graph model via pykeen
    :param model_name: name of the model, i.e. transE, PairRE
    :param train_path: path to training file
    :param test_path: path to test file
    :return: pykeen pipeline result
    """
    result = pipeline(
        training=train_path,
        testing=test_path,
        model=model_name,
        # epochs=128,  # short epochs for testing - you should go higher TODO: non-functional, and random seed
    )
    result.save_to_directory(f'model/{model_name}')
    return result


def prepare_embd(result, dataset, lm_model_name, annotation, kg_model_name, on_tiny=False):
    """
    Function to prepare all embeddings combination for further use
    :param result: pykeen pipeline result
    :param dataset: ["training", "development", "test"]
    :param lm_model_name: name of the language model
    :param annotation: annotation style
    :param kg_model_name: name of the knowledge graph model
    :param on_tiny: boolean, if process on tiny set
    :return: None
    """
    dataset = dataset + "_tiny" if on_tiny else dataset
    model_ann_name = lm_model_name + "_" + annotation + "_tiny_ft" \
        if on_tiny else lm_model_name + "_" + annotation + "_ft"
    df = load_from_bin(dataset)
    entity_embedding_tensor = result.model.entity_representations[0](indices=None).cpu().detach().numpy()
    logging.info(f"sheep: {entity_embedding_tensor.shape}")
    embd_shape = entity_embedding_tensor.shape[1]
    pos_count = 0
    neg_count = 0
    arg1_list, arg2_list = [], []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        arg1 = row["Ent1"]
        arg2 = row["Ent2"]
        try:
            arg1_embd = get_kg_embd(arg1, entity_embedding_tensor, result)
            pos_count += 1
            arg1_list.append(arg1_embd)
        except KeyError:
            arg1_list.append(np.zeros(embd_shape))
            neg_count += 1
        try:
            arg2_embd = get_kg_embd(arg2, entity_embedding_tensor, result)
            pos_count += 1
            arg2_list.append(arg2_embd)
        except KeyError:
            arg2_list.append(np.zeros(embd_shape))
            neg_count += 1
    logging.info(f"pos: {pos_count}, neg: {neg_count}, ratio: {pos_count / (pos_count + neg_count)}")

    df[f"arg1_embd_{kg_model_name}"] = arg1_list
    df[f"arg2_embd_{kg_model_name}"] = arg2_list
    df[f"combi_embd_{kg_model_name}_{model_ann_name}"] = df.apply(
        lambda x: concatenate_embd(x[f"sent_embd_{model_ann_name}"],
                                   x[f"arg1_embd_{kg_model_name}"],
                                   x[f"arg2_embd_{kg_model_name}"]),
        axis=1)
    save_to_bin(df, dataset)


def main():
    kg_models = ["TransE", "PairRE"]
    models = [
        # "bert-base-uncased",
        "allenai/scibert_scivocab_uncased",
        "dmis-lab/biobert-base-cased-v1.1",
    ]
    annotations = ["raw", "sci", "bio"]
    datasets = ["training", "development", "test"]
    for kg_model in kg_models:
        result = train_kg_model(kg_model)
        for dataset in datasets:
            for model in models:
                for annotation in annotations:
                    prepare_embd(result, dataset, model, annotation, kg_model)
                    prepare_embd(result, dataset, model, annotation, kg_model, on_tiny=True)


if __name__ == '__main__':
    main()
