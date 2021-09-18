import torch
import logging
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import os
from util import create_tensor_dataset, format_time, get_sent_embd, load_from_bin, save_to_bin
import time
import numpy as np
from sklearn import metrics
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def lm_fine_tune(args: dict) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logging.info(f'Use the GPU: {torch.cuda.get_device_name(0)}')
    else:
        logging.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    model_name = args["model_name"]
    annotation = args["annotation"]
    dataset = args["dataset"]
    max_length = args["max_length"]
    on_tiny = args["on_tiny"]
    batch_size = args["batch_size"]
    epochs = args["epochs"]
    seed_val = args["seed_val"]
    device_ids = args["device_ids"]
    local_files_only = args["local_files_only"]

    model_path = os.path.join("model", model_name)
    tokenizer = BertTokenizer.from_pretrained(model_path,
                                              do_lower_case=True,
                                              local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path,
                                                          output_hidden_states=True,
                                                          local_files_only=True)
    model.to(device)
    pred_dataset = create_tensor_dataset(tokenizer,
                                         dataset=dataset,
                                         max_length=max_length,
                                         annotation=annotation,
                                         on_tiny=on_tiny)
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=batch_size)
    model.eval()



