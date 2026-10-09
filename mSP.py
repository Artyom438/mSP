import pickle
import numpy as np
import pandas as pd

import os
import sys
import time
from glob import glob

from lib.paths import *
sys.path.insert(0, 'FExt')
from FExt.GenFeatures import GenFeatures

def ElapsedTime(startTime, message='Total running time: '):
    endTime = time.time()
    totalTime = endTime - startTime
    hours = totalTime // 3600
    minutes = (totalTime - hours * 3600) // 60
    seconds = totalTime - hours * 3600 - minutes * 60
    return f'{message}{totalTime}   {hours}:{minutes}:{seconds}'

def FeatureReduction(features, singleValues, reduction, scaler=None, postScler=None):
    features = np.delete(features, singleValues, axis=1)

    if scaler is not None:
        features = scaler.transform(features)
    
    if reduction is not None:
        features = reduction.transform(features)
    else:
        features = features[:, -192:]

    if postScler is not None:
        features = postScler.transform(features)
    
    return features

if __name__ == '__main__':
    if sys.argv[1] in {'h', '-h', 'help', '--help'} or len(sys.argv) != 3:
        print('Usage one of two: \npython3 mSP.py input.fasta output.csv \npython3 mSP.py input.npy output.csv')
        sys.exit()
        
    startTime = time.time()
    if len(sys.argv[1]) > 5 and sys.argv[1][-6:].lower() == '.fasta':
        print(f'Begin feature extraction for {sys.argv[1]}')
        os.chdir('FExt')
        featuresPath = os.path.join('features', os.path.split(sys.argv[1])[-1][:-6] + '.npy')
        fastaPath = os.path.join('..', sys.argv[1])
        GenFeatures(fastaPath, featuresPath)
        os.chdir('..')
        featuresPath = os.path.join('FExt', featuresPath)
        print(ElapsedTime(startTime, message='Feature extraction time: '))
    else:
        featuresPath = sys.argv[1]
    
    features = np.load(featuresPath)
    print(f'Loaded from {featuresPath} features shape: {features.shape}')
    
    singleValues = np.load('lib/singleValues.npy')
    metaFeatures = []
    for i, path in enumerate(baseModelsPaths):
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
            print(f'For model {i} load feature reduction {reductionPath}')
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

    predTable = pd.DataFrame(probs, columns=classes)
    predTable['Class'] = preds
    predTable.to_csv(sys.argv[2], index=False)
    print(f'Predictions saved in {sys.argv[2]}')
    print(ElapsedTime(startTime))
