import torch
import logging
import pandas as pd
import numpy as np
from util import create_tensor_dataset, check_gpu_mem, flat_accuracy, format_time
from util import get_train_stats, save_train_stats, save_model
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import random
import time
import torch.nn as nn

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
    max_length = args["max_length"]
    on_tiny = args["on_tiny"]
    batch_size = args["batch_size"]
    epochs = args["epochs"]
    seed_val = args["seed_val"]
    device_ids = args["device_ids"]

    # load the BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    # load data into dataloader
    train_dataset = create_tensor_dataset(tokenizer,
                                          dataset="training",
                                          max_length=max_length,
                                          annotation=annotation,
                                          on_tiny=on_tiny)
    val_dataset = create_tensor_dataset(tokenizer,
                                        dataset="development",
                                        max_length=max_length,
                                        annotation=annotation,
                                        on_tiny=on_tiny)
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        model_name,  # Use the 12-layer BERT model
        num_labels=14,  # 14 labels, including "NONE" relation
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    # data parallel
    model = nn.DataParallel(model, device_ids=device_ids)
    logging.info(f"GPU memory info:\n{check_gpu_mem()}")

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    for epoch_i in range(0, epochs):
        # Perform one full pass over the training set.
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logging.info('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                logging.info('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # Check GPU memory for the first couple steps.
            if step < 2:
                logging.info('Step {:} GPU Memory Use:'.format(step))
                logging.info('Before forward-pass:\n{}'.format(check_gpu_mem()))
            model.zero_grad()
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)
            loss = result.loss
            logits = result.logits

            # Report GPU memory use for the first couple steps.
            if step < 2:
                logging.info('After forward-pass:\n{}'.format(check_gpu_mem()))
            total_train_loss += loss.item()
            loss.backward()
            if step < 2:
                logging.info('After gradient calculation:\n{}'.format(check_gpu_mem()))
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        logging.info("Average training loss: {0:.2f}".format(avg_train_loss))
        logging.info("Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        logging.info("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)
                loss = result.loss
                logits = result.logits
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        logging.info("Accuracy: {0:.2f}".format(avg_val_accuracy))
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)
        logging.info("Validation Loss: {0:.2f}".format(avg_val_loss))
        logging.info("Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    logging.info("Training complete!")
    df_stats = get_train_stats(training_stats)
    logging.info(f"training status:\n{df_stats}")
    model_file_name = f"{model_name}_{annotation}_tiny_ft" if on_tiny else f"{model_name}_{annotation}_ft"
    save_train_stats(df_stats, model_file_name)
    save_model(model_file_name, model, tokenizer)


def main():
    # args on local machine
    # args = {
    #     "model_name": "bert-base-uncased",
    #     "annotation": "raw",
    #     "max_length": 192,
    #     "on_tiny": True,
    #     "batch_size": 8,
    #     "epochs": 2,
    #     "seed_val": 1024,
    #     "device_ids": [0],
    # }

    # args on GPU server
    args = {
        "model_name": "bert-base-uncased",
        "annotation": "raw",
        "max_length": 192,
        "on_tiny": False,
        "batch_size": 32,
        "epochs": 4,
        "seed_val": 1024,
        "device_ids": [0, 1],
    }

    models = [
        "bert-base-uncased",
        "allenai/scibert_scivocab_uncased",
        "dmis-lab/biobert-base-cased-v1.1",
    ]
    annotations = ["raw", "sci", "bio"]
    for model in models:
        for annotation in annotations:
            args["model_name"] = model
            args["annotation"] = annotation
            lm_fine_tune(args)


if __name__ == '__main__':
    main()
