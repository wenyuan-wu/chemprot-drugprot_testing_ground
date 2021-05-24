from transformers import *
import numpy as np

model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

doc = """
Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity 
ligand,[(125)I]iodoazidosalmeterol.
"""

tokens = tokenizer.tokenize(doc)

print(len(tokens))
print(tokens)

pt_batch = tokenizer(
    doc,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}")

for sents in pt_batch["input_ids"]:
    print(sents)
    for ids in sents.tolist():
        print(f"id: {ids}, token: {tokenizer.convert_ids_to_tokens(ids)}")

pt_outputs = model(**pt_batch)

print(pt_outputs.last_hidden_state[0, 0, :])
print(pt_outputs.pooler_output.size())
