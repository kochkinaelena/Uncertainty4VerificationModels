from loading_model import load_model, predict_on_data, get_sorted_replies
import pickle
from convert2trees import branch2tree
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from read_fullPHEME_data import read_fullPHEME
import json

#%%

folds = read_fullPHEME()

for foldid in list(folds.keys()):
    
    model, params = load_model(foldid)
    
    
    filepath = "saved_datafullPHEME/"+foldid+"/preprocessed_expanded_folds_fullPHEME.json"
    with open (filepath, 'rb') as f:
        expanded_folds = pickle.load(f)
    
    print (foldid)
    uncertainties_in_time = {}
    for ii, treeid in enumerate(list(expanded_folds[foldid].keys())):

        if ii%5==0:
            print (ii)
        
        tree = expanded_folds[foldid][treeid]
        
        for conv in folds[foldid]:
             if conv['id']==treeid:
                 src = conv['source']
                 
        
        sorted_replies = [src] + get_sorted_replies(folds, foldid, treeid)  

        aleatoric_in_time = []
        variation_ratio_in_time = []
        variance_in_time = []
        entropy_in_time = []
        softmax_in_time = []
        predictions_in_time = []
#        subtree_predicitons = []
        
        for sub in list(tree.keys()):
            
            subtree = tree[sub]
            if subtree['features']!=[]:
                subtree_features = pad_sequences(subtree['features'], maxlen=None,
                                                 dtype='float32',
                                                 padding='post',
                                                 truncating='post', value=0.)
            
                subtree_labels = np.asarray(subtree['veracity_labels'])
                subtree_ids = np.asarray(subtree['ids'])
            
                predictions = predict_on_data(model,params, subtree_features, subtree_labels, num_classes=3, verbose=False)
                
                if len(subtree_features)>1:
                    tree_results_test = branch2tree(subtree_ids, predictions)
                    aleatoric_in_time.append(float(tree_results_test[treeid]['aleatoric_uncertainty_avg']))
                    variation_ratio_in_time.append(float(tree_results_test[treeid]['variation_ratio_avg']))
                    variance_in_time.append(float(tree_results_test[treeid]['variance_1_avg']))
                    entropy_in_time.append(float(tree_results_test[treeid]['predictive_entropy_avg']))
                    softmax_in_time.append(float(np.max(tree_results_test[treeid]['avg_softmax_from_samples_avg'])))
                    predictions_in_time.append(float(tree_results_test[treeid]['tree_prediction_from_avg_softmax']))
                    label = tree_results_test[treeid]['true_label']
                    
                elif len(subtree_features)==1:
                    aleatoric_in_time.append(float(predictions['aleatoric_uncertainty']))
                    variation_ratio_in_time.append(float(predictions['variation_ratio']))
                    variance_in_time.append(float(np.mean(predictions['variance_1'])))
                    entropy_in_time.append(float(predictions['predictive_entropy']))
                    softmax_in_time.append(float(np.max(predictions['avg_softmax_from_samples'])))
                    predictions_in_time.append(float(predictions['prediction_from_avg_softmax']))
                    label = float(predictions['true_label'])
            else:
                print(treeid)
                print(sub)
                
#%%
        uncertainties_in_time[treeid] = {}
        uncertainties_in_time[treeid]['aleatoric_in_time'] = aleatoric_in_time
        uncertainties_in_time[treeid]['variation_ratio_in_time'] = variation_ratio_in_time 
        uncertainties_in_time[treeid]['variance_in_time'] = variance_in_time
        uncertainties_in_time[treeid]['entropy_in_time'] = entropy_in_time  
        uncertainties_in_time[treeid]['sorted_replies'] = sorted_replies
        
        uncertainties_in_time[treeid]['softmax_in_time'] = softmax_in_time
        uncertainties_in_time[treeid]['predictions_in_time'] = predictions_in_time
        uncertainties_in_time[treeid]['label'] = float(label)
       
    save_file = "uncertainties_in_time"+str(foldid)+".json"
    
    with open(save_file, 'w') as f:
        json.dump(uncertainties_in_time,f)     
    print ("saved fold")











