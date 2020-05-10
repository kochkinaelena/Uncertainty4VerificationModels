import os
import numpy as np
import pickle
from sklearn.metrics import f1_score, accuracy_score

#%%

#with open (filepath+"/bestparams_avgw2v.txt", 'rb') as f:
#        params = pickle.load(f)       
#with open (filepath+"/trials_avgw2v.txt", 'rb') as f:
#        trials = pickle.load(f)
        
#best_trial_id = trials.best_trial["tid"]
#best_trial_loss = trials.best_trial["result"]["loss"]

#%%

filepath = "output"
folds = ['0','1','2','3', '4']
true_labels = []
predicted_labels = []
eval_info = []
p = []
# predictions
for fo in folds:
    true_labels_fold = []
    predicted_labels_fold = []
    
    with open (os.path.join(filepath,'predictions'+fo+'.pkl'), 'rb') as f:
        predictions_fold = pickle.load(f)
        
    p.append(predictions_fold)
    
    with open (os.path.join(filepath,'eval_info'+fo+'.pkl'), 'rb') as f:
        eval_info.append(pickle.load(f))

    for tree in list(predictions_fold['test']['tree_results'].values()):
         true_labels.append(tree['true_label'])
         predicted_labels.append(tree['tree_prediction_from_avg_softmax'])
         true_labels_fold.append(tree['true_label'])
         predicted_labels_fold.append(tree['tree_prediction_from_avg_softmax'])
         
    print(fo)
    print(accuracy_score(true_labels_fold, predicted_labels_fold))
    print(f1_score(true_labels_fold, predicted_labels_fold, average='macro'))
    print(eval_info[-1]['test']['rmse_softmax'])

#with open (os.path.join(filepath,'eval_info'+folds[4]+'.pkl'), 'rb') as f:
#    eval_info = pickle.load(f)
         
#%%

print(accuracy_score(true_labels, predicted_labels))
print(f1_score(true_labels, predicted_labels, average='macro'))