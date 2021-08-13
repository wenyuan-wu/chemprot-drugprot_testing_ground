import torch
import logging
import pandas as pd
from drugprot_preprocess import load_from_file

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

