import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    # datefmt='%d-%b-%y %H:%M:%S'
                    )


def get_sent_embd(tokenizer, model, text, max_length):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
    layer_i = 0

    print("Number of batches:", len(hidden_states[layer_i]))
    batch_i = 0

    print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    token_i = 0

    print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
    # encoded_dict = tokenizer.encode_plus(
    #     text,  # Sentence to encode.
    #     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    #     max_length=max_length,  # Pad & truncate all sentences.
    #     padding='max_length',
    #     truncation=True,
    #     return_attention_mask=True,  # Construct attn. masks.
    #     return_tensors='pt',  # Return pytorch tensors.
    # )
    pass


def main():
    model_name = "bert_tiny_none"
    model_path = os.path.join("model", model_name)
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path,
                                                          output_hidden_states=True,
                                                          local_files_only=True,
                                                          )
    text = "Here is the sentence I want embeddings for."
    get_sent_embd(tokenizer, model, text, max_length=192)


if __name__ == '__main__':
    main()
