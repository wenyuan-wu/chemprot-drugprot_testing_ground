import numpy as np
import pykeen.utils
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import pandas as pd
from tqdm import tqdm
from util import load_from_bin, save_to_bin
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


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
    save_to_bin(result.training.entity_to_id, f"{model_name}_entity_to_id")
    return result


def main():
    kg_models = ["TransE", "PairRE"]
    train_path = os.path.join(os.getcwd(), "data/drugprot_preprocessed/kg/drugprot_kg_train_combi.tsv")
    test_path = os.path.join(os.getcwd(), "data/drugprot_preprocessed/kg/drugprot_kg_dev.tsv")
    for kg_model in kg_models:
        result = train_kg_model(kg_model, train_path, test_path)


if __name__ == '__main__':
    main()
