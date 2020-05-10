import os
import numpy as np
import pickle
from sklearn.metrics import f1_score, accuracy_score
#%%
filepath = ""

folds = ['ebola-essien','ferguson', 'gurlitt',
         'ottawashooting', 'prince-toronto', 'putinmissing', 
         'sydneysiege', 'charliehebdo','germanwings-crash' ]


true_labels = []
predicted_labels = []

# predictions
for f in folds:
    true_labels_fold = []
    predicted_labels_fold = []
    with open (os.path.join(filepath,'predictions'+f+'.pkl'), 'rb') as fin:
        predictions_fold = pickle.load(fin)

    for tree in list(predictions_fold['test']['tree_results'].values()):
         true_labels.append(tree['true_label'])
         predicted_labels.append(tree['tree_prediction_from_avg_softmax'])
         true_labels_fold.append(tree['true_label'])
         predicted_labels_fold.append(tree['tree_prediction_from_avg_softmax'])
    print(f)
    print(accuracy_score(true_labels_fold, predicted_labels_fold))
    print(f1_score(true_labels_fold, predicted_labels_fold, average='macro'))

print(accuracy_score(true_labels, predicted_labels))
print(f1_score(true_labels, predicted_labels, average='macro'))
