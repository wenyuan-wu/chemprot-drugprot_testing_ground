from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-base-Pubmed_PMC")
model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-base-Pubmed_PMC")

sentence = "Identification of APC2 , a homologue of the adenomatous polyposis coli tumour suppressor ."
text = "ncbi_ner: " + sentence + " </s>"

encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

model = model.to("cuda")

outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    early_stopping=True
)

for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # line = tokenizer.decode(output)
    print(line)

# doc = """
# Probing the salmeterol binding site on the beta 2-adrenergic receptor using a novel photoaffinity
# ligand,[(125)I]iodoazidosalmeterol.
# """

# tokens = tokenizer.tokenize(doc)
#
# print(len(tokens))
# print(tokens)
#
# pt_batch = tokenizer(
#     doc,
#     padding=True,
#     truncation=True,
#     max_length=512,
#     return_tensors="pt"
# )
#
# for key, value in pt_batch.items():
#     print(f"{key}: {value.numpy().tolist()}")
#
# for sents in pt_batch["input_ids"]:
#     print(sents)
#     for ids in sents.tolist():
#         print(f"id: {ids}, token: {tokenizer.convert_ids_to_tokens(ids)}")
#
# pt_outputs = model(**pt_batch)
#
# print(pt_outputs.last_hidden_state[0, 0, :])
# print(pt_outputs.pooler_output.size())
