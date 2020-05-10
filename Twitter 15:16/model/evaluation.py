from rmse import rmse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
#%%

def eval_timeline(predictions):
    eval_info = {}
    
    Y_true = predictions['true_label']
    Y_pred = predictions['prediction_from_softmax_raw']
    
    confidence = [np.max(predictions['softmax_raw'][i]) for i in range(len(Y_true))]
    
    eval_info['accuracy'] = accuracy_score(Y_true, Y_pred)
    
    eval_info['macroF'] = f1_score(Y_true, Y_pred, average='macro',labels=[0,1,2])
    
    eval_info['rmse'] = rmse(Y_true, Y_pred, confidence)
    
    return eval_info



def eval_branches(tree_results):
    
    eval_info = {}
    
    Y_true = [tree_results[i]['true_label']  for i in list(tree_results.keys())]
    Y_pred = [tree_results[i]['tree_prediction_from_avg_softmax']  for i in list(tree_results.keys())]
    
    confidence_softmax = [np.max(tree_results[i]['softmax_raw_avg'])  for i in list(tree_results.keys())]

    eval_info['accuracy'] = accuracy_score(Y_true, Y_pred)
    
    eval_info['macroF'] = f1_score(Y_true, Y_pred, average='macro',labels=[0,1,2])
    
    eval_info['rmse_softmax'] = rmse(Y_true, Y_pred, confidence_softmax)
    
    return eval_info