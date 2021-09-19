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

if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    logging.info(f'Use the GPU: {torch.cuda.get_device_name(0)}')

else:
    logging.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model_name = "bert_raw_tiny"
model_path = os.path.join("model", model_name)

# load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path,
                                                      output_hidden_states=True,
                                                      local_files_only=True)
model.to(device)

batch_size = 8
dev_dataset = create_tensor_dataset("dev_raw_tiny", tokenizer)

# Create a sequential sampler--no need to randomize the order!
dev_sampler = SequentialSampler(dev_dataset)

# Create the data loader.
dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size)

# Predict labels for all test set examples.

# print('Predicting labels for {:,} test(dev) comments...'.format(len(dev_dataloader)))

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions, true_labels = [], []
sent_embd_list = []

# Measure elapsed time.
t0 = time.time()

# Predict
for (step, batch) in enumerate(dev_dataloader):

    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)

        # Report progress.
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dev_dataloader), elapsed))

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store the compute graph, saving memory
    # and speeding up prediction
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

print('    DONE.')

# Combine the results across the batches.
predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
sentence_embeddings = np.concatenate(sent_embd_list, axis=0)

# Take the highest scoring output as the predicted label.
predicted_labels = np.argmax(predictions, axis=1)

print('`predictions` has shape', predictions.shape)
print('`predicted_labels` has shape', predicted_labels.shape)
print('`sentences embeddings` has shape', sentence_embeddings.shape)
# Reduce printing precision for legibility.
np.set_printoptions(precision=2)

print("Predicted:", str(predicted_labels[0:10]))
print("  Correct:", str(true_labels[0:10]))
print("first sentence embd:", sentence_embeddings[0])

# Use the F1 metric to score our classifier's performance on the test set.
score = metrics.f1_score(true_labels, predicted_labels, average='macro')

# Print the F1 score!
print('F1 score: {:.3}'.format(score))

df = load_from_bin("dev_raw_tiny")
idx_to_label_dict = load_from_bin("idx_to_label_dict_raw")
print(df.columns)
df["label_pred"] = predicted_labels
df["relation_pred"] = df["label_pred"].map(idx_to_label_dict)
df["sent_embd"] = [x for x in tqdm(sentence_embeddings)]
# counter = 0
for idx, row in df.iterrows():
    # row["sent_embd"] = sentence_embeddings[counter]
    # counter += 1
    print(idx)
    print(row["sent_embd"].shape)

print(df.columns)
save_to_bin(df, "dev_raw_tiny_lm")