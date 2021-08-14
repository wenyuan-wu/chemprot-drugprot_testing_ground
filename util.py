from os.path import join
import pandas as pd
import pickle
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def save_to_bin(tmp_dict, file_name):
    file_path = join("data", "drugprot_preprocessed", "bin", file_name)
    infile = open(file_path, 'wb')
    pickle.dump(tmp_dict, infile)
    logging.info(f"File saved in {file_path}")
    infile.close()


def load_from_bin(file_name):
    file_path = join("data", "drugprot_preprocessed", "bin", file_name)
    outfile = open(file_path, 'rb')
    logging.info(f"Loading file from {file_path}")
    tmp_dict = pickle.load(outfile)
    outfile.close()
    return tmp_dict

