import torch
import logging
import pandas as pd
from util import load_from_bin

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )

if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    logging.info(f'Use the GPU: {torch.cuda.get_device_name(0)}')

else:
    logging.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

df_train = load_from_bin("train_drop_0.85")
df_dev = load_from_bin("dev_drop_0.85")
# print(df_train)

