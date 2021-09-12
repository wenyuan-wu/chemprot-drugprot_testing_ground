from pykeen.pipeline import pipeline
import pandas as pd
from drugprot_preprocess import get_df_from_data
from tqdm import tqdm
from util import load_from_bin
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def get_kd_embd(entity, embd_tensor, result):
    idx = result.training.entity_to_id[entity]
    return embd_tensor[idx]


def main():
    train_path = "data/drugprot_preprocessed/kg/drugprot_kg_train_combi.tsv"
    test_path = "data/drugprot_preprocessed/kg/drugprot_kg_dev.tsv"
    result = pipeline(
        training=train_path,
        testing=test_path,
        model='PairRE',

        # epochs=128,  # short epochs for testing - you should go higher
    )
    result.save_to_directory('model/test_drugprot_transe')
    df_train = pd.DataFrame.from_dict(load_from_bin("train_org"), orient="index")
    entity_embedding_tensor = result.model.entity_representations[0](indices=None).cpu().detach().numpy()
    logging.info(f"sheep: {entity_embedding_tensor.shape}")
    pos_count = 0
    neg_count = 0
    for idx, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
        arg1 = list(row["ent_dict"].values())[0]["Text"].lower()
        arg2 = list(row["ent_dict"].values())[1]["Text"].lower()
        try:
            get_kd_embd(arg1, entity_embedding_tensor, result)
            pos_count += 1
            # print(f"entity: {entity}\nembd: {get_kd_embd(entity, result)}")
        except KeyError:
            # print("ERROR!")
            # print(entity)
            neg_count += 1

        try:
            get_kd_embd(arg2, entity_embedding_tensor, result)
            pos_count += 1
            # print(f"entity: {entity}\nembd: {get_kd_embd(entity, result)}")
        except KeyError:
            # print("ERROR!")
            # print(entity)
            neg_count += 1
    logging.info(f"pos: {pos_count}, neg: {neg_count}, ratio: {pos_count / (pos_count + neg_count)}")
    # entity = "epinephrine"
    # print(f"entity: {entity}\nembd: {get_kd_embd(entity, result)}")


if __name__ == '__main__':
    main()

