import pickle
import numpy as np
import pandas as pd

import os
import sys
from glob import glob

from imblearn.metrics import specificity_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, matthews_corrcoef, f1_score, precision_score, recall_score

from lib.paths import *

def FeatureReduction(features, singleValues, reduction, scaler=None, postScler=None):
    features = features.drop(['label', 'SampleName'] + list(singleValues), axis=1).values
    # print(f'Features shape after remove columns: {features.shape}')

    if scaler is not None:
        features = scaler.transform(features)
        # print(f'Features shape after feature scaling: {features.shape}')
        # print(f'After feature scaling features mean {np.mean(features)}, std {np.std(features)}')
    
    if reduction is not None:
        features = reduction.transform(features)
    else:
        features = features[:, -192:]
    # print(f'Features shape after feature reduction: {features.shape}')

    if postScler is not None:
        features = postScler.transform(features)
        # print(f'Features shape after post feature scaling: {features.shape}')
        # print(f'After post feature scaling features mean {np.mean(features)}, std {np.std(features)}')
    
    return features

if __name__ == '__main__':
    if sys.argv[1] in {'h', '-h', 'help', '--help'} or len(sys.argv) != 3:
        print('Usage: \npython3 mSP.py input.fasta output.csv')
        sys.exit()
        
    features = pd.read_csv(sys.argv[1])
    print(f'Loaded from {sys.argv[1]} features shape: {features.shape}')
    
    singleValues = np.load('lib/singleValues.npy')
    metaFeatures = []
    for path in baseModelsPaths:
        with open(path, 'rb') as f:
            baseModel = pickle.load(f)
        
        pathData = os.path.split(path)[-1].split('_')[:3]
        if '_PCA' in path:
            prePath = 'lib/scalers/PCA_' + os.path.split(path)[-1][:3] + '.pkl'
            with open(prePath, 'rb') as f:
                preScaler = pickle.load(f)
            postScaler = None
        else:
            preScaler = None
            
            postPath = f'lib/scalers/{pathData[2]}_{pathData[0]}_{pathData[1]}.pkl'
            with open(postPath, 'rb') as f:
                postScaler = pickle.load(f)
        
        if '_Z-Curve_' in path:
            reduction = None
        else:
            reductionPath = f'lib/FEng/{pathData[2]}_{pathData[0]}_{pathData[1]}.pkl'
            with open(reductionPath, 'rb') as f:
                reduction = pickle.load(f)
        
        reducedFeatures = FeatureReduction(features, singleValues, reduction, scaler=preScaler, postScler=postScaler)
        metaFeatures.append(baseModel.predict_proba(reducedFeatures)[:, 0])
    
    metaFeatures = np.transpose(np.stack(metaFeatures))
    print(f'metaFeatures.shape: {metaFeatures.shape}')

    with open('lib/metaModels/metaTrain50_CatBoostClassifier_bestModelAllTrain.pickle', 'rb') as f:
        metaModel = pickle.load(f)
    probs = metaModel.predict_proba(metaFeatures)
    classes = np.array(['Cytoplasm', 'Endoplasmic reticulum', 'Extracellular region', 'Mitochondria', 'Nucleus'])
    preds = classes[np.argmax(probs, axis=1)]
    # print(preds.shape)

    predTable = pd.DataFrame(probs, columns=classes)
    predTable['Class'] = preds
    predTable.to_csv(sys.argv[2])
    print(f'Predictions saved in {sys.argv[2]}')
