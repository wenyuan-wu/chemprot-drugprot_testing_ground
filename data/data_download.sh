wget https://zenodo.org/record/5119892/files/drugprot-training-development-test-background.zip
unzip drugprot-training-development-test-background.zip
rm drugprot-training-development-test-background.zip
mv drugprot-gs-training-development drugprot

# rename files to fix typo in file names
mv drugprot/training/drugprot_training_abstracs.tsv drugprot/training/drugprot_training_abstracts.tsv
mv drugprot/development/drugprot_development_abstracs.tsv drugprot/development/drugprot_development_abstracts.tsv

echo "drugprot data downloaded"
