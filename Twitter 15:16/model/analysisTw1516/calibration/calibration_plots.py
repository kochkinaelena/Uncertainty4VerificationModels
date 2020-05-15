#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
random.seed(364)
from cal_methods import HistogramBinning
from cal_methods import cal_probs
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MinMaxScaler

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
        
#        if (not np.isnan(tree_results[k]['predictive_entropy_avg'])):
#            entropy = tree_results[k]['predictive_entropy_avg']
#        else: 
#            entropy = 0
       
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
    
folds = ['1','2','3','4']    
#filepath = "Twitter15/output_dev_set_separate/output/"
filepath = "Twitter16/output_dev_set_separate/output/"
files = os.listdir(filepath)

softmax_raw_avgall_probs_test = []
softmax_raw_avgall_y_test = []
softmax_raw_avgall_cal_probs_test = []
softmax_raw_avgall_correct_test = []

variation_ratio_avgall_probs_test = []
variation_ratio_avgall_y_test = []
variation_ratio_avgall_cal_probs_test = []
variation_ratio_avgall_correct_test = []

aleatoric_uncertainty_avgall_probs_test = []
aleatoric_uncertainty_avgall_y_test = []
aleatoric_uncertainty_avgall_cal_probs_test = []
aleatoric_uncertainty_avgall_correct_test = []
uncertainty_name = 'aleatoric_uncertainty_avg'

for fold in folds:
    print (fold)
    with open(filepath+"predictions"+fold+".pkl", 'rb') as f:
        predictions = pickle.load(f)

    aleatoric_uncertainty_avguncert_val, aleatoric_uncertainty_avgy_val, aleatoric_uncertainty_avgy_val_correct, _ = pred2feats(predictions, 'dev', uncertainty_name)
    aleatoric_uncertainty_avguncert_test, aleatoric_uncertainty_avgy_test, aleatoric_uncertainty_avgy_test_correct, _ = pred2feats(predictions, 'test', uncertainty_name)
    
    
    scaler = MinMaxScaler()
    aleatoric_uncertainty_avguncert_val_scaled = scaler.fit_transform(np.asarray(aleatoric_uncertainty_avguncert_val).reshape(-1, 1))
    aleatoric_uncertainty_avguncert_val_scaled = aleatoric_uncertainty_avguncert_val_scaled.reshape(1, -1)[0]
    
    for i, val in enumerate(aleatoric_uncertainty_avguncert_val_scaled):
        if val>1:
            print ('nono')
            aleatoric_uncertainty_avguncert_val_scaled[i]=1
    
    
    aleatoric_uncertainty_avguncert_test_scaled = scaler.fit_transform(np.asarray(aleatoric_uncertainty_avguncert_test).reshape(-1, 1))
    aleatoric_uncertainty_avguncert_test_scaled = aleatoric_uncertainty_avguncert_test_scaled.reshape(1, -1)[0]
    
    for i, val in enumerate(aleatoric_uncertainty_avguncert_test_scaled):
        if val>1.0:
            aleatoric_uncertainty_avguncert_test_scaled[i]=1.0

    aleatoric_uncertainty_avgprobs_val = np.array(1 - aleatoric_uncertainty_avguncert_val_scaled)
    aleatoric_uncertainty_avgprobs_test = np.array(1- aleatoric_uncertainty_avguncert_test_scaled)
    

    aleatoric_uncertainty_avgcal_probs_test = cal_probs(HistogramBinning, aleatoric_uncertainty_avgprobs_val, aleatoric_uncertainty_avgy_val_correct, aleatoric_uncertainty_avgprobs_test,{'M':15})

    aleatoric_uncertainty_avgall_probs_test.extend(aleatoric_uncertainty_avgprobs_test)
    aleatoric_uncertainty_avgall_y_test.extend(aleatoric_uncertainty_avgy_test)
    aleatoric_uncertainty_avgall_cal_probs_test.extend(aleatoric_uncertainty_avgcal_probs_test)
    aleatoric_uncertainty_avgall_correct_test.extend(aleatoric_uncertainty_avgy_test_correct)
    
    uncertainty_name = 'softmax_raw_avg'
    
    softmax_raw_avguncert_val, softmax_raw_avgy_val, softmax_raw_avgy_val_correct, _ = pred2feats(predictions, 'dev', uncertainty_name)
    softmax_raw_avguncert_test, softmax_raw_avgy_test, softmax_raw_avgy_test_correct, _ = pred2feats(predictions, 'test', uncertainty_name)
    
    softmax_raw_avgprobs_val = softmax_raw_avguncert_val
    softmax_raw_avgprobs_test = softmax_raw_avguncert_test
    
    softmax_raw_avgcal_probs_test = cal_probs(HistogramBinning, softmax_raw_avgprobs_val, softmax_raw_avgy_val_correct, softmax_raw_avgprobs_test,{'M':15})
    
    softmax_raw_avgall_probs_test.extend(softmax_raw_avgprobs_test)
    softmax_raw_avgall_y_test.extend(softmax_raw_avgy_test)
    softmax_raw_avgall_cal_probs_test.extend(softmax_raw_avgcal_probs_test)
    softmax_raw_avgall_correct_test.extend(softmax_raw_avgy_test_correct)
    
    uncertainty_name = 'variation_ratio_avg'
    
    uncert_val, y_val, y_val_correct, _ = pred2feats(predictions, 'dev', uncertainty_name)
    uncert_test, y_test, y_test_correct, _ = pred2feats(predictions, 'test', uncertainty_name)
    
    variation_ratio_avgprobs_val = np.array(1 - uncert_val)
    variation_ratio_avgprobs_test = np.array(1- uncert_test)
    
    variation_ratio_avgcal_probs_test = cal_probs(HistogramBinning, variation_ratio_avgprobs_val, y_val_correct, variation_ratio_avgprobs_test,{'M':15})
    
    variation_ratio_avgall_probs_test.extend(variation_ratio_avgprobs_test)
    variation_ratio_avgall_y_test.extend(y_test)
    variation_ratio_avgall_cal_probs_test.extend(variation_ratio_avgcal_probs_test)
    variation_ratio_avgall_correct_test.extend(y_test_correct)
     
#%%

from eval import ECE_score

aleatoric_uncertainty_avgece = ECE_score(aleatoric_uncertainty_avgall_correct_test, aleatoric_uncertainty_avgall_probs_test, bin_size = 0.1) 
aleatoric_uncertainty_avgece2 = ECE_score(aleatoric_uncertainty_avgall_correct_test, aleatoric_uncertainty_avgall_cal_probs_test, bin_size = 0.1)  

softmax_raw_avgece = ECE_score(softmax_raw_avgall_correct_test, softmax_raw_avgall_probs_test, bin_size = 0.1) 
softmax_raw_avgece2 = ECE_score(softmax_raw_avgall_correct_test, softmax_raw_avgall_cal_probs_test, bin_size = 0.1)  


variation_ratio_avgece = ECE_score(variation_ratio_avgall_correct_test, variation_ratio_avgall_probs_test, bin_size = 0.1) 
variation_ratio_avgece2 = ECE_score(variation_ratio_avgall_correct_test, variation_ratio_avgall_cal_probs_test, bin_size = 0.1)    
    
#%%    


aleatoric_uncertainty_avgcorrect = aleatoric_uncertainty_avgall_correct_test
aleatoric_uncertainty_avgconfs = aleatoric_uncertainty_avgall_probs_test
aleatoric_uncertainty_avgconfs2 = aleatoric_uncertainty_avgall_cal_probs_test

softmax_raw_avgcorrect = softmax_raw_avgall_correct_test
softmax_raw_avgconfs = softmax_raw_avgall_probs_test
softmax_raw_avgconfs2 = softmax_raw_avgall_cal_probs_test

variation_ratio_avgcorrect = variation_ratio_avgall_correct_test
variation_ratio_avgconfs = variation_ratio_avgall_probs_test
variation_ratio_avgconfs2 = variation_ratio_avgall_cal_probs_test

# Plot calibration plots


#name='variance_output_'+folder+'dptest'
plt.rcParams.update({'font.size': 25})
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:",lw=4, label="Perfectly calibrated")
#
#fraction_of_positives, mean_predicted_value = calibration_curve(aleatoric_uncertainty_avgcorrect, aleatoric_uncertainty_avgconfs, n_bins=10)
#name='Aleatoric (No C)'
#ax1.plot(mean_predicted_value, fraction_of_positives, "s-", color='blue',linewidth=4,
#             label="%s" % (name, ))
#
#
#fraction_of_positives2, mean_predicted_value2 = calibration_curve(aleatoric_uncertainty_avgcorrect, aleatoric_uncertainty_avgconfs2, n_bins=10)
#name2 = 'Aleatoric (HB)'
#ax1.plot(mean_predicted_value2, fraction_of_positives2, "s--", color='red',linewidth=4,
#             label="%s" % (name2, ))

fraction_of_positives3, mean_predicted_value3 = calibration_curve(variation_ratio_avgcorrect, variation_ratio_avgconfs, n_bins=10)
name3='Variation ratio (No C)'
ax1.plot(mean_predicted_value3, fraction_of_positives3, "s-", color='blue',linewidth=4,
             label="%s" % (name3, ))


fraction_of_positives4, mean_predicted_value4 = calibration_curve(variation_ratio_avgcorrect, variation_ratio_avgconfs2, n_bins=10)
name4 = 'Variation ratio (HB)'
ax1.plot(mean_predicted_value4, fraction_of_positives4, "s--", color='red',linewidth=4,
             label="%s" % (name4, ))

#fraction_of_positives5, mean_predicted_value5 = calibration_curve(softmax_raw_avgcorrect, softmax_raw_avgconfs, n_bins=10)
#name5='Softmax (No C)'
#ax1.plot(mean_predicted_value5, fraction_of_positives5, "s-",color='blue',linewidth=4,
#             label="%s" % (name5, ))
#
#
#fraction_of_positives6, mean_predicted_value6 = calibration_curve(softmax_raw_avgcorrect, softmax_raw_avgconfs2, n_bins=10)
#name6 = 'Softmax (HB)'
#ax1.plot(mean_predicted_value6, fraction_of_positives6, "s--",color='red',linewidth=4,
#             label="%s" % (name6, ))

#
##
#ax2.hist(aleatoric_uncertainty_avgconfs, range=(0, 1), bins=10, label=name, color='blue', linewidth=4,
#             histtype="step")
#
#ax2.hist(aleatoric_uncertainty_avgconfs2, range=(0, 1), bins=10, label=name2, color='red',linewidth=4,
#             histtype="step")


ax2.hist(variation_ratio_avgconfs, range=(0, 1), bins=10, label=name3, color='blue',linewidth=4,
             histtype="step")

ax2.hist(variation_ratio_avgconfs2, range=(0, 1), bins=10, label=name4, color='red',linewidth=4,
             histtype="step")

##
#ax2.hist(softmax_raw_avgconfs, range=(0, 1), bins=10, label=name5,color='blue',linewidth=4,
#             histtype="step")
#
#ax2.hist(softmax_raw_avgconfs2, range=(0, 1), bins=10, label=name6,color='red',linewidth=4,
#             histtype="step")

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
#ax1.legend(loc="lower right")
ax1.legend(loc="upper left")
ax1.set_title('Twitter 16')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
#ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()
fig.savefig(name3+'.png')

