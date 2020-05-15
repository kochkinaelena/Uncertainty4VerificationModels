#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
import random
random.seed(364)
from cal_methods import HistogramBinning
from cal_methods import cal_probs
from sklearn.preprocessing import MinMaxScaler
#%%
# load eval and predictions
filepath = "results/PHEME/output_dev_set_separate/output/"
files = os.listdir(filepath)
#%%

def pred2feats(predictions, whichset, uncertainty_name):
    
    X = []
    Y = []
    ids = []
    y_val_correct = []
    if whichset=='test' or whichset=='dev':
        tree_results = predictions[whichset]['tree_results']
    if whichset=='train':
        tree_results = {}
        for fold_dict in predictions[whichset]:
            tree_results.update(fold_dict['tree_results'])
    
    
    for k in sorted(list(tree_results.keys())):
        
        features = []
        
        if uncertainty_name=='softmax_raw_avg':
            features.append(np.max(tree_results[k][uncertainty_name])) 
        else:
            features.append(tree_results[k][uncertainty_name]) 

        X.extend(np.asarray(features).astype(float))
        ids.append(k)
        
        Y.append(tree_results[k]['true_label'])
        
        y_val_correct.append(np.argmax(tree_results[k]['softmax_raw_avg'])==tree_results[k]['true_label'])
        

    return np.asarray(X),np.asarray(Y), np.array(y_val_correct, dtype="int"), ids
#%%
folds = ['ebola-essien','ferguson', 'gurlitt',
         'ottawashooting', 'prince-toronto', 'putinmissing', 
         'sydneysiege', 'germanwings-crash' ] #  'charliehebdo'


all_probs_test = []
all_y_test = []
all_cal_probs_test = []
all_correct_test = []

uncertainty_name = 'aleatoric_uncertainty_avg'

for fold in folds:
    print (fold)
    with open(filepath+"predictions"+fold+".pkl", 'rb') as f:
        predictions = pickle.load(f)
        
    uncert_val, y_val, y_val_correct, _ = pred2feats(predictions, 'dev', uncertainty_name)
    uncert_test, y_test, y_test_correct, _ = pred2feats(predictions, 'test', uncertainty_name)
    
    if uncertainty_name== "aleatoric_uncertainty_avg" or uncertainty_name== "predictive_entropy_avg" or uncertainty_name== "variance_1_avg": 
    
        scaler = MinMaxScaler()
        uncert_val_scaled = scaler.fit_transform(np.asarray(uncert_val).reshape(-1, 1))
        uncert_val_scaled = uncert_val_scaled.reshape(1, -1)[0]
        
        for i, val in enumerate(uncert_val_scaled):
            if val>1:
                uncert_val_scaled[i]=1
        
        
        uncert_test_scaled = scaler.fit_transform(np.asarray(uncert_test).reshape(-1, 1))
        uncert_test_scaled = uncert_test_scaled.reshape(1, -1)[0]
        
        for i, val in enumerate(uncert_test_scaled):
            if val>1.0:
                uncert_test_scaled[i]=1.0
    
        probs_val = np.array(1 - uncert_val_scaled)
        probs_test = np.array(1- uncert_test_scaled)
    
    elif uncertainty_name== "variation_ratio_avg":
    
        probs_val = np.array(1 - uncert_val)
        probs_test = np.array(1- uncert_test)
        
    elif uncertainty_name== "softmax_raw_avg":
        
        probs_val = np.array(uncert_val)
        probs_test = np.array(uncert_test)
        
        
    cal_probs_test = cal_probs(HistogramBinning, probs_val, y_val_correct, probs_test,{'M':15})

    all_probs_test.extend(probs_test)
    all_y_test.extend(y_test)
    all_cal_probs_test.extend(cal_probs_test)
    all_correct_test.extend(y_test_correct)
#%%

from eval import ECE_score

ece = ECE_score(all_correct_test, all_probs_test, bin_size = 0.1) 
ece2 = ECE_score(all_correct_test, all_cal_probs_test, bin_size = 0.1)    
    