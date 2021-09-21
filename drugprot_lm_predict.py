import torch
import logging
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import os
from util import create_tensor_dataset, format_time, get_sent_embd, load_from_bin, save_to_bin, check_gpu
import time
import numpy as np
from tqdm import tqdm
import torch.nn as nn

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def lm_predict(args: dict) -> None:
    """
    Predict labels on desired dataset via preloaded models (local files)
    :param args: dictionary contains essential parameters
    :return: None, dataset files will be altered accordingly
    """
    device = check_gpu()
    model_name = args["model_name"]
    annotation = args["annotation"]
    dataset = args["dataset"]
    max_length = args["max_length"]
    on_tiny = args["on_tiny"]
    batch_size = args["batch_size"]
    device_ids = args["device_ids"]
    local_files_only = args["local_files_only"]
    model_file_name = f"{model_name}_{annotation}_tiny_ft" if on_tiny else f"{model_name}_{annotation}_ft"
    model_path = os.path.join("model", model_file_name)
    tokenizer = BertTokenizer.from_pretrained(model_path,
                                              do_lower_case=True,
                                              local_files_only=local_files_only)
    model = BertForSequenceClassification.from_pretrained(model_path,
                                                          output_hidden_states=True,
                                                          local_files_only=local_files_only)
    # data parallel TODO: fix data parallel issue on multiple GPUs
    # model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    pred_dataset = create_tensor_dataset(tokenizer,
                                         dataset=dataset,
                                         max_length=max_length,
                                         annotation=annotation,
                                         on_tiny=on_tiny)
    pred_sampler = SequentialSampler(pred_dataset)
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=batch_size)
    model.eval()
    # Tracking variables
    predictions, true_labels = [], []
    sent_embd_list = []
    # Measure elapsed time.
    t0 = time.time()

    for (step, batch) in enumerate(pred_dataloader):
        batch = tuple(t.to(device) for t in batch)
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(pred_dataloader), elapsed))
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        sentence_embeddings = get_sent_embd(hidden_states)
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
        sent_embd_list.append(sentence_embeddings)
    logging.info('DONE.')
    # Combine the results across the batches.
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    sentence_embeddings = np.concatenate(sent_embd_list, axis=0)
    # Take the highest scoring output as the predicted label.
    predicted_labels = np.argmax(predictions, axis=1)

    logging.info(f'`predictions` has shape {predictions.shape}')
    logging.info(f'`predicted_labels` has shape {predicted_labels.shape}')
    logging.info(f'`sentences embeddings` has shape {sentence_embeddings.shape}')
    # handle tiny dataset
    dataset = dataset + "_tiny" if on_tiny else dataset
    df = load_from_bin(dataset)
    idx_to_label_dict = load_from_bin("idx_to_label_dict")
    df[f"label_{model_file_name}"] = predicted_labels
    df[f"relation_{model_file_name}"] = df[f"label_{model_file_name}"].map(idx_to_label_dict)
    df[f"sent_embd_{model_file_name}"] = [x for x in tqdm(sentence_embeddings)]
    save_to_bin(df, dataset)


def main():
    # # args on GPU server
    args = {
        "model_name": "bert-base-uncased",
        "annotation": "raw",
        "dataset": "development",
        "max_length": 192,
        "on_tiny": False,
        "batch_size": 32,
        "device_ids": [0, 1],
        "local_files_only": True,
    }

    # args on local machine
    # args = {
    #     "model_name": "bert-base-uncased",
    #     "annotation": "raw",
    #     "dataset": "development",
    #     "max_length": 192,
    #     "on_tiny": True,
    #     "batch_size": 8,
    #     "device_ids": [0, 1],
    #     "local_files_only": True,
    # }

    models = [
        # "bert-base-uncased",
        "allenai/scibert_scivocab_uncased",
        "dmis-lab/biobert-base-cased-v1.1",
    ]
    annotations = ["raw", "sci", "bio"]
    datasets = ["training", "development", "test"]

    for dataset in datasets:
        for model in models:
            for annotation in annotations:
                logging.info(f"dataset: {dataset}, model: {model}, annotation: {annotation}")
                args["model_name"] = model
                args["annotation"] = annotation
                args["dataset"] = dataset
                lm_predict(args)


if __name__ == '__main__':
    main()
