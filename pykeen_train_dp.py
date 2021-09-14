import numpy as np
from pykeen.pipeline import pipeline
import pandas as pd
from drugprot_preprocess import get_df_from_data
from tqdm import tqdm
from util import load_from_bin, save_to_bin
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def get_kd_embd(entity, embd_tensor, result):
    idx = result.training.entity_to_id[entity]
    return embd_tensor[idx]


def concatenate_embd(arr1, arr2, arr3):
    return np.concatenate((arr1, arr2, arr3), axis=None)


def main():
    train_path = "data/drugprot_preprocessed/kg/drugprot_kg_train_combi.tsv"
    test_path = "data/drugprot_preprocessed/kg/drugprot_kg_dev.tsv"
    result = pipeline(
        training=train_path,
        testing=test_path,
        model='PairRE',
        # epochs=128,  # short epochs for testing - you should go higher
    )
    result.save_to_directory('model/drugprot_pairre')
    df = load_from_bin("train_sci_lm")
    entity_embedding_tensor = result.model.entity_representations[0](indices=None).cpu().detach().numpy()
    logging.info(f"sheep: {entity_embedding_tensor.shape}")
    pos_count = 0
    neg_count = 0
    arg1_list, arg2_list = [], []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        arg1 = row["Ent1"]
        arg2 = row["Ent2"]
        try:
            arg1_embd = get_kd_embd(arg1, entity_embedding_tensor, result)
            pos_count += 1
            arg1_list.append(arg1_embd)
            # print(arg1_embd.shape)
            # print(f"entity: {arg1}\nembd: {get_kd_embd(arg1, entity_embedding_tensor, result)}")
        except KeyError:
            # print("ERROR!")
            # print(entity)
            arg1_list.append(np.zeros(200))
            neg_count += 1
        try:
            arg2_embd = get_kd_embd(arg2, entity_embedding_tensor, result)
            pos_count += 1
            arg2_list.append(arg2_embd)
            # print(f"entity: {arg2}\nembd: {get_kd_embd(arg2, entity_embedding_tensor, result)}")
        except KeyError:
            # print("ERROR!")
            # print(entity)
            arg2_list.append(np.zeros(200))
            neg_count += 1
    logging.info(f"pos: {pos_count}, neg: {neg_count}, ratio: {pos_count / (pos_count + neg_count)}")
    df["arg1_embd"] = arg1_list
    df["arg2_embd"] = arg2_list
    df["con_embd"] = df.apply(lambda x: concatenate_embd(x["sent_embd"],
                                                         x["arg1_embd"],
                                                         x["arg2_embd"]), axis=1)
    save_to_bin(df, "train_sci_lm_kg")

    ##########################################################################test##################################
    df = load_from_bin("test_sci_lm")
    entity_embedding_tensor = result.model.entity_representations[0](indices=None).cpu().detach().numpy()
    logging.info(f"sheep: {entity_embedding_tensor.shape}")
    pos_count = 0
    neg_count = 0
    arg1_list, arg2_list = [], []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        arg1 = row["Ent1"]
        arg2 = row["Ent2"]
        try:
            arg1_embd = get_kd_embd(arg1, entity_embedding_tensor, result)
            pos_count += 1
            arg1_list.append(arg1_embd)
            # print(arg1_embd.shape)
            # print(f"entity: {arg1}\nembd: {get_kd_embd(arg1, entity_embedding_tensor, result)}")
        except KeyError:
            # print("ERROR!")
            # print(entity)
            arg1_list.append(np.zeros(200))
            neg_count += 1
        try:
            arg2_embd = get_kd_embd(arg2, entity_embedding_tensor, result)
            pos_count += 1
            arg2_list.append(arg2_embd)
            # print(f"entity: {arg2}\nembd: {get_kd_embd(arg2, entity_embedding_tensor, result)}")
        except KeyError:
            # print("ERROR!")
            # print(entity)
            arg2_list.append(np.zeros(200))
            neg_count += 1
    logging.info(f"pos: {pos_count}, neg: {neg_count}, ratio: {pos_count / (pos_count + neg_count)}")
    df["arg1_embd"] = arg1_list
    df["arg2_embd"] = arg2_list
    df["con_embd"] = df.apply(lambda x: concatenate_embd(x["sent_embd"],
                                                         x["arg1_embd"],
                                                         x["arg2_embd"]), axis=1)
    save_to_bin(df, "test_sci_lm_kg")


if __name__ == '__main__':
    main()
