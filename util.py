from os.path import join
import pandas as pd
import numpy as np
import pickle
import logging
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, random_split
import os
import csv
import datetime

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def save_to_bin(tmp_dict, file_name):
    # dir_path = join("data", "drugprot_preprocessed")
    # os.mkdir(dir_path)
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


def sent_len_dist(sents, tokenizer):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    # Record the length of each sequence (in terms of BERT tokens).
    lengths = []
    logging.info('Tokenizing sentences...')

    for sen in tqdm(sents):
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sen,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # max_length = 512,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
            verbose=False  # hide warnings
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
        # Record the non-truncated length.
        lengths.append(len(encoded_sent))

    logging.info('Min length: {:,} tokens'.format(min(lengths)))
    logging.info('Max length: {:,} tokens'.format(max(lengths)))
    logging.info('Median length: {:,} tokens'.format(np.median(lengths)))

    num_over = 0
    for length in lengths:
        # Tally if it's over 512.
        if length > 512:
            num_over += 1
    logging.info('{:,} of {:,} sentences will be truncated ({:.2%})'
                 .format(num_over, len(lengths), float(num_over) / float(len(lengths))))


def create_tensor_dataset(data_name, tokenizer):
    data_df = load_from_bin(data_name)
    data_df = data_df.reset_index()
    data_df.rename(columns={'index': "sent_id"}, inplace=True)
    data_df.relation = pd.Categorical(data_df.relation)
    data_df["label"] = data_df.relation.cat.codes

    # Get the lists of sentences and their labels.
    sentences = data_df.text_raw.values
    labels = data_df.label.values

    # get sentence length distribution information
    # sent_len_dist(sentences, tokenizer)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in tqdm(sentences):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset


def check_gpu_mem():
    """
    Uses Nvidia's SMI tool to check the current GPU memory usage.
    Reported values are in "MiB". 1 MiB = 2^20 bytes = 1,048,576 bytes.
    """
    # Run the command line tool and get the results.
    buf = os.popen('nvidia-smi --query-gpu=memory.total,memory.used --format=csv')
    # Use csv module to read and parse the result.
    reader = csv.reader(buf, delimiter=',')
    # Use a pandas table just for nice formatting.
    df = pd.DataFrame(reader)
    # Use the first row as the column headers.
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row
    df.columns = new_header  # set the header row as the df header

    return df


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# class CustomTextDataset(Dataset):
#     def __init__(self, dataframe):
#         self.labels = dataframe.relation
#         self.text = dataframe.text_raw
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         label = self.labels[idx]
#         text = self.text[idx]
#         sample = {"Text": text, "Class": label}
#         return sample
#
#
# TD = CustomTextDataset(df_train)
# print('\nFirst iteration of data set: ', next(iter(TD)), '\n')
#
#
# # Print entire data set
# # print('Entire data set: ', list(DataLoader(TD)), '\n')
# DL = torch.utils.data.DataLoader(TD, batch_size=256, shuffle = True)
# for (idx, batch) in enumerate(DL):
#     print(batch['Text'])
#     print(batch['Class'])
#
#
# def collate_batch(batch):
#     text_tensor = BertTokenizer.encode()
#     word_tensor = torch.tensor([[1.], [0.], [45.]])
#     label_tensor = torch.tensor([[1.]])
#
#     text_list, classes = [], []
#     for (_text, _class) in batch:
#         encoded_dict = tokenizer.encode_plus(
#             _text,  # Sentence to encode.
#             add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#             max_length=64,  # Pad & truncate all sentences.
#             pad_to_max_length=True,
#             return_attention_mask=True,  # Construct attn. masks.
#             return_tensors='pt',  # Return pytorch tensors.
#         )
#
#         text_list.append(word_tensor)
#         classes.append(label_tensor)
#     text = torch.cat(text_list)
#     classes = torch.tensor(classes)
#     return text, classes
