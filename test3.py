import torch
# from transformers import BertTokenizer, BertForSequenceClassification
import logging
import os

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
    file_path = os.path.join(os.getcwd(), "drugprot_kg_train_combi.tsv")
    print(file_path)
    path_test(file_path)


if __name__ == '__main__':
    main()
