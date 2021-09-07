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
import seaborn as sns
import matplotlib.pyplot as plt

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


def sent_len_dist(sents, tokenizer, max_length=192):
    logging.info(f"max length: {max_length}")
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
        # Tally if it's over 192.
        if length > max_length:
            num_over += 1
    logging.info('{:,} of {:,} sentences will be truncated ({:.2%})'
                 .format(num_over, len(lengths), float(num_over) / float(len(lengths))))


def create_tensor_dataset(data_name, tokenizer, max_length=192, annotation="raw"):
    logging.info(f"max length: {max_length}")
    data_df = load_from_bin(data_name)
    data_df = data_df.reset_index()
    data_df.rename(columns={'index': "sent_id"}, inplace=True)
    # TODO: fix cat issues
    # data_df.relation = pd.Categorical(data_df.relation)
    # data_df["label"] = data_df.relation.cat.codes

    # Get the lists of sentences and their labels.
    if annotation == "raw":
        sentences = data_df.text_raw.values
    elif annotation == "scibert":
        sentences = data_df.text_scibert.values
    elif annotation == "biobert":
        sentences = data_df.text_biobert.values
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
            max_length=max_length,  # Pad & truncate all sentences.
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


def get_train_stats(train_stats):
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=train_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '50px')])])

    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    # Display the table.
    return df_stats


def save_train_stats(df_stats, model_name):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    output_dir = join("model", model_name)
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = f"train_stat_{datetime.datetime.now()}.png"
    output_path = join(output_dir, file_name)
    plt.savefig(output_path)


def save_model(model_name, model, tokenizer):
    output_dir = join("model", model_name)
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))


def sent_annotation(gene, chem, sent, soi, annotation):
    sent_list = []
    # update ent_dict to focus on the only two entities involved in the relation
    ent_dict = {gene["Entity #"]: gene, chem["Entity #"]: chem}
    srt_list = sorted(ent_dict.items(), key=lambda x: x[1]["Start"], reverse=False)

    if annotation == "scibert":
        """
        idx_1: start of the first entity
        char_1: annotation for the start character of the first entity, depending on the type
        idx_2: end the first entity
        char_2: annotation for the end character of the first entity, depending on the type
        idx_3, char_3, idx_4, char_4: same deal to the second character
        """
        idx_1 = srt_list[0][1]["Start"] - soi
        char_1 = "<< " if srt_list[0][1]["Type"].startswith("CHEM") else "[[ "
        idx_2 = srt_list[0][1]["End"] - soi - 1
        char_2 = " >>" if srt_list[0][1]["Type"].startswith("CHEM") else " ]]"
        idx_3 = srt_list[1][1]["Start"] - soi
        char_3 = "<< " if srt_list[1][1]["Type"].startswith("CHEM") else "[[ "
        idx_4 = srt_list[1][1]["End"] - soi - 1
        char_4 = " >>" if srt_list[1][1]["Type"].startswith("CHEM") else " ]]"
        for idx, char in enumerate(list(sent)):
            if idx == idx_1:
                char = char_1 + char
            elif idx == idx_2:
                char = char + char_2
            elif idx == idx_3:
                char = char_3 + char
            elif idx == idx_4:
                char = char + char_4
            sent_list.append(char)

        return "".join(sent_list)

    elif annotation == "biobert":
        """
        idx_1: start of the first entity
        idx_2: end the first entity
        idx_range_1: index span of the first entity
        char_1: annotation of the first entity, depending on the type
        idx_3, idx_4, idx_range_2, char_2: same deal to the second character
        """
        idx_1 = srt_list[0][1]["Start"] - soi
        idx_2 = srt_list[0][1]["End"] - soi - 1
        char_1 = "$CHEMICAL#" if srt_list[0][1]["Type"].startswith("CHEM") else "$GENE#"
        idx_range_1 = list(range(idx_1 + 1, idx_2 + 1))
        idx_3 = srt_list[1][1]["Start"] - soi
        idx_4 = srt_list[1][1]["End"] - soi - 1
        idx_range_2 = list(range(idx_3 + 1, idx_4 + 1))
        char_2 = "$CHEMICAL#" if srt_list[1][1]["Type"].startswith("CHEM") else "$GENE#"
        for idx, char in enumerate(list(sent)):
            if idx == idx_1:
                char = char_1
            elif idx in idx_range_1:
                char = ""
            elif idx == idx_3:
                char = char_2
            elif idx in idx_range_2:
                char = ""
            sent_list.append(char)
        return "".join(sent_list)
