This model, data, and code are for the paper "A Comparative Study of Models for mRNA Subcellular Localization Prediction Using a Combined Dataset Benchmark" https://link.springer.com/chapter/10.1007/978-3-032-39382-1_16

Don't forget download files from lfs:
git lfs pull

Import conda enviroment:
conda env create -f mSP.yml

Activate conda enviroment:
conda activate mSP

Extract features and predict on test set:
python3 mSP.py data/test.fasta predictions/testPred.csv



Or predict on test set if features already extracted:
python3 mSP.py FExt/features/test.npy predictions/testPred.csv

In this example predictions will be saved in predictions/testPred.csv
You can view It in check_mSP_predictions.ipynb or open with LibreOffice Calc or Excel

For check software and debugging small data/testExample.fasta can be used
python3 mSP.py data/testExample.fasta predictions/testExamplePred.csv

For help enter:
python3 mSP.py -h

Class labels:
{'Cytoplasm': 0,
 'Endoplasmic reticulum': 1,
 'Extracellular region': 2,
 'Mitochondria': 3,
 'Nucleus': 4}
