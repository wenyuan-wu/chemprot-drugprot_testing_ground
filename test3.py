import torch
# from transformers import BertTokenizer, BertForSequenceClassification
import logging
import os
from util import load_from_bin

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def path_test(path):
    with open(path) as infile:
        for line in infile:
            print(line)
            print('!' * 25)
            break


def main():
    vdict = load_from_bin("PairRE_entity_to_id")
    for k, v in vdict.items():
        print(k, v)


if __name__ == '__main__':
    main()
