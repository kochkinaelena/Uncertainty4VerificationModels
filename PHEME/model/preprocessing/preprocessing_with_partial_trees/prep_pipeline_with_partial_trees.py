"""
This is outer preprocessing file

To run:
    
python prep_pipeline.py

Main function has parameters that can be changed:
    
dataset ('RumEv' or 'fullPHEME') 
and feats ('text' or 'SemEval')

"""
from read_fullPHEME_data import read_fullPHEME
from read_RumEv_data import read_RumEv
from transform_feature_dict import transform_feature_dict
from extract_thread_features import extract_thread_features_incl_response
import help_prep_functions
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from copy import deepcopy
from add_partial_trees import add_partial_folds
import json
import pickle
#%%


def convert_label(label):
    if label == "true":
        return(0)
    elif label == "false":
        return(1)
    elif label == "unverified":
        return(2)
    else:
        print(label)

        
dataset='fullPHEME'
feature_set=['avgw2v']
path = 'saved_data'+dataset
folds = {}
if dataset == 'RumEv':
    folds = read_RumEv()
else:
    folds = read_fullPHEME()

newfolds = add_partial_folds(folds)    
    
help_prep_functions.loadW2vModel()


#%%    
for fold in list(newfolds.keys()):
    print(fold)

    preprocessed_expanded_folds = {}
    preprocessed_expanded_folds[fold] = {}
    for tree in list(newfolds[fold].keys()):
         preprocessed_expanded_folds[fold][tree] = {}
         subtree_dict = newfolds[fold][tree]
         for subtree_num in list(subtree_dict.keys()):
             preprocessed_expanded_folds[fold][tree][subtree_num] = {}
             
             subtree = subtree_dict[subtree_num]
             
             thread_feature_dict = extract_thread_features_incl_response(subtree)

             thread_features_array, thread_stance_labels, branches = transform_feature_dict(
                                   thread_feature_dict, subtree,
                                   feature_set=feature_set)
             
             preprocessed_expanded_folds[fold][tree][subtree_num]['features'] = deepcopy(thread_features_array)
             preprocessed_expanded_folds[fold][tree][subtree_num]['tweet_ids'] = branches
             preprocessed_expanded_folds[fold][tree][subtree_num]['fold_stance_labels'] = thread_stance_labels
             preprocessed_expanded_folds[fold][tree][subtree_num]['veracity_labels'] = []
             preprocessed_expanded_folds[fold][tree][subtree_num]['ids'] = []
             for i in range(len(thread_features_array)):
                  preprocessed_expanded_folds[fold][tree][subtree_num]['veracity_labels'].append(convert_label(subtree['veracity']))
                  preprocessed_expanded_folds[fold][tree][subtree_num]['ids'].append(subtree['id'])
    


    filepath = path+'/'+str(fold)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    #%%
    with open(filepath+'/preprocessed_expanded_folds_'+dataset+'.json', 'wb') as fp:
        pickle.dump(preprocessed_expanded_folds, fp)
            
