# ChemProt Testing Ground
Testing ground for [task 5](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/) from BioCreative VI
> Task 5: Text mining chemical-protein interactions (CHEMPROT)
> 
> The aim of the CHEMPROT task of BioCreative VI is to promote the development and evaluation of systems that are able 
> to automatically detect in running text (PubMed abstracts) relations between chemical compounds/drug and genes/proteins. 
> We will therefore release a manually annotated corpus, the CHEMPROT corpus, where domain experts have exhaustively labeled:
> (a) all chemical and gene mentions, and 
> (b) all binary relationships between them corresponding to a specific set of biologically relevant relation types 
> (CHEMPROT relation classes).

## Working Environment
- Ubuntu 20.04
- CUDA 11.2
- PyTorch 1.8.1+cu111

## Data Structure
```
data/
├── chemprot_development
│   ├── chemprot_development_abstracts.tsv
│   ├── chemprot_development_entities.tsv
│   ├── chemprot_development_gold_standard.tsv
│   ├── chemprot_development_relations.tsv
│   └── Readme.pdf
├── chemprot_sample
│   ├── chemprot_sample_abstracts.tsv
│   ├── chemprot_sample_entities.tsv
│   ├── chemprot_sample_gold_standard.tsv
│   ├── chemprot_sample_predictions_eval.txt
│   ├── chemprot_sample_predictions.tsv
│   ├── chemprot_sample_relations.tsv
│   ├── guidelines
│   │   ├── CEM_guidelines.pdf
│   │   ├── CHEMPROT_guidelines_v6.pdf
│   │   └── GPRO_guidelines.pdf
│   └── Readme.pdf
├── chemprot_test_gs
│   ├── chemprot_test_abstracts_gs.tsv
│   ├── chemprot_test_entities_gs.tsv
│   ├── chemprot_test_gold_standard.tsv
│   ├── chemprot_test_relations_gs.tsv
│   └── readme_test_gs.pdf
└── chemprot_training
    ├── chemprot_training_abstracts.tsv
    ├── chemprot_training_entities.tsv
    ├── chemprot_training_gold_standard.tsv
    ├── chemprot_training_relations.tsv
    └── Readme.pdf

5 directories, 25 files
```
