import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import os
from copy import deepcopy
import random
random.seed(364)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
#%%

def branch2tree(ids_test, branch_results):

    trees = np.unique(ids_test)

    
    result = {}

    for tree in trees:
        
        result[tree] = {}
        
        treeindx = np.where(ids_test == tree)[0]
  
        branch = {}
        result[tree]['branches'] = []
        for i in treeindx:
            for key in list(branch_results.keys()):
                branch[key] = branch_results[key][i]
                
            result[tree]['branches'].append(deepcopy(branch))
            
     
        list_of_keys = ['softmax_raw', 'avg_softmax_from_samples',
                        'aleatoric_uncertainty', 'predictive_entropy', 
                        'variance_3', 'variance_1', 'variation_ratio']
        for key in list(list_of_keys):
            result[tree][key+'raw_avg'] = [branch_results[key][i] for i in treeindx]
            result[tree][key+'_avg'] = np.mean(result[tree][key+'raw_avg'],axis=0)
        
        result[tree]['true_label'] = [branch_results['true_label'][i] for i in treeindx][0]
        
        result[tree]['tree_prediction_from_avg_softmax'] = np.argmax(result[tree]['avg_softmax_from_samples_avg'])
        result[tree]['tree_prediction_from_avg_avg_softmax'] = np.argmax(result[tree]['softmax_raw_avg'])
        result[tree]['branch_predictions'] = [branch_results['prediction_from_softmax_raw'][i] for i in treeindx]
        unique, counts = np.unique(result[tree]['branch_predictions'], return_counts=True)
        result[tree]['tree_prediction_majority_vote'] = unique[np.argmax(counts)]
        
        result[tree]['tree_prediction_weighted_avg_aleatoric'] = [result[tree]['softmax_rawraw_avg'][i]*result[tree]['aleatoric_uncertaintyraw_avg'][i]
                                                        for i in range(len(result[tree]['aleatoric_uncertaintyraw_avg']))]
        result[tree]['tree_prediction_weighted_avg_aleatoric'] = np.argmax(np.mean(result[tree]['tree_prediction_weighted_avg_aleatoric'], axis=0))
        
        result[tree]['tree_prediction_weighted_avg_epistemic'] = [result[tree]['softmax_rawraw_avg'][i]*result[tree]['variance_1raw_avg'][i]
                                                        for i in range(len(result[tree]['variance_1raw_avg']))]
        result[tree]['tree_prediction_weighted_avg_epistemic'] = np.argmax(np.mean(result[tree]['tree_prediction_weighted_avg_epistemic'], axis=0))
     
        result[tree]['tree_prediction_weighted_avg_entropy'] = [result[tree]['softmax_rawraw_avg'][i]*result[tree]['predictive_entropyraw_avg'][i]
                                                        for i in range(len(result[tree]['predictive_entropyraw_avg']))]
        result[tree]['tree_prediction_weighted_avg_entropy'] = np.argmax(np.mean(result[tree]['tree_prediction_weighted_avg_entropy'], axis=0))
        
        
    return result


#%%
def pred2feats(predictions, whichset):
    
    X = []
    Y = []
    ids = []
    if whichset=='test' or whichset=='dev':
        tree_results = predictions[whichset]['tree_results']
    if whichset=='train':
        tree_results = {}
        for fold_dict in predictions[whichset]:
            tree_results.update(fold_dict['tree_results'])
    
    
    for k in sorted(list(tree_results.keys())):
        
        if (not np.isnan(tree_results[k]['predictive_entropy_avg'])):
            entropy = tree_results[k]['predictive_entropy_avg']
        else: 
            entropy = 0
       
        features = [
                    tree_results[k]['aleatoric_uncertainty_avg'],
#                    tree_results[k]['predictive_entropy_avg'],
                    tree_results[k]['variance_1_avg'],
                    tree_results[k]['variation_ratio_avg'],
                    tree_results[k]['tree_prediction_from_avg_avg_softmax'],
#                    np.max(tree_results[k]['softmax_raw_avg'])
                    ]
        
        features.extend(tree_results[k]['softmax_raw_avg'])
        features.append(entropy)
        X.append(features)
        ids.append(k)
        if tree_results[k]['true_label']==tree_results[k]['tree_prediction_from_avg_avg_softmax']:
            Y.append(1)
        else:
            Y.append(0)
        
    
    return X,Y,ids


#%%
# load eval and predictions
filepath = "output"
files = os.listdir(filepath)
#%%
# convert predictions 

folds = ['ebola-essien','ferguson', 'gurlitt',
         'ottawashooting', 'prince-toronto', 'putinmissing', 
         'sydneysiege', 'germanwings-crash' ] #  'charliehebdo'


all_true = []
all_pred_SVM = []
all_pred_RF = []
all_test_ids = []

veracity_true = []
veracity_pred = []
veracity_ids = []
aleatoric_uncertainty = []
predictive_entropy = []
variation_ratio = []
softmax = []
variance = []



for fold in folds:
    print (fold)
    with open(filepath+"predictions"+fold+".pkl", 'rb') as f:
        predictions = pickle.load(f)
        
    prediction = predictions['test']['tree_results']
    
    vtrue = []
    vpred = []
    
    for item in sorted(list(prediction.keys())):
        
         
         vtrue.append(prediction[item]['true_label'])
         vpred.append(prediction[item]['tree_prediction_from_avg_avg_softmax'])
         
         aleatoric_uncertainty.append(prediction[item]['aleatoric_uncertainty_avg'])
         if  (not np.isnan(prediction[item]['predictive_entropy_avg'])):
             predictive_entropy.append(prediction[item]['predictive_entropy_avg'])
         else:
             predictive_entropy.append(0)
         softmax.append(np.max(prediction[item]['softmax_raw_avg']))
         variance.append(prediction[item]['variance_1_avg'])
         variation_ratio.append(prediction[item]['variation_ratio_avg'])
         veracity_ids.append(item)
        
    veracity_true.extend(vtrue)
    veracity_pred.extend(vpred)
    
    print("Fold Veracity Accuracy ",accuracy_score(vtrue, vpred))
    print("Fold Veracity Macro F ",f1_score(vtrue, vpred, average='macro')) 

#    X_train,Y_train = pred2feats(predictions, 'train')
    X_dev,Y_dev,ids_dev = pred2feats(predictions, 'dev')
    X_test,Y_test, ids_test = pred2feats(predictions, 'test')
    
#    cw = {0:1 ,1:3} #'balanced'
    
    clf = SVC(kernel='linear',gamma='auto',random_state=0, class_weight='balanced') # kernel='linear', default is 'rbf'
    clf.fit(X_dev, Y_dev) 
    Y_pred_SVM = clf.predict(X_test)
    
    
    print("SVM Accuracy: ", accuracy_score(Y_test, Y_pred_SVM))
    print("SVM Macro F: ", f1_score(Y_test, Y_pred_SVM, average='macro'))
    
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0, class_weight='balanced')
    clf.fit(X_dev, Y_dev) 
    Y_pred_RF = clf.predict(X_test)
    
# TODO:     PLAY WITH PROBABILITY THRESHOLD TO GET BEST PRECISION!!    
    
    print("RF Accuracy: ",accuracy_score(Y_test, Y_pred_RF))
    print("RF MacroF: ",f1_score(Y_test, Y_pred_RF, average='macro'))
    
    all_true.extend(Y_test)
    all_pred_SVM.extend(Y_pred_SVM)
    all_pred_RF.extend(Y_pred_RF)
    all_test_ids.extend(ids_test)

print("All Veracity Accuracy ",accuracy_score(veracity_true, veracity_pred))
print("All Veracity Macro F ",f1_score(veracity_true, veracity_pred, average='macro')) 

print("All SVM Accuracy: ",accuracy_score(all_true, all_pred_SVM))
print("All SVM MacroF: ",f1_score(all_true, all_pred_SVM, average='macro'))  
print("All SVM Precision: ",precision_score(all_true, all_pred_SVM))  
print("All SVM Recall: ",recall_score(all_true, all_pred_SVM))  
    
    
print("All RF Accuracy: ",accuracy_score(all_true, all_pred_RF))
print("All RF MacroF: ",f1_score(all_true, all_pred_RF, average='macro')) 
print("All RF Precision: ",precision_score(all_true, all_pred_RF))  
print("All RF Recall: ",recall_score(all_true, all_pred_RF))   


#%%

assert (all_test_ids==veracity_ids)

#%%

# supervised rejection part

veracity_true_cp = deepcopy(veracity_true)
veracity_pred_cp = deepcopy(veracity_pred)

print ("What did we need to predict ? ", np.unique(all_true, return_counts=True))
print ("What did we predict (SVM)? ", np.unique(all_pred_SVM, return_counts=True))
print ("What did we predict (RF)? ", np.unique(all_pred_RF, return_counts=True))

for i,item in reversed(list(enumerate(veracity_true))):
    if all_pred_SVM[i]==0:
        del (veracity_true_cp[i])
        del (veracity_pred_cp[i])
        
print (len(veracity_true_cp))
print (len(veracity_pred_cp))
        
print("Veracity Accuracy after SVM rejection",accuracy_score(veracity_true_cp, veracity_pred_cp))
print("Veracity Macro F after SVM rejection",f1_score(veracity_true_cp, veracity_pred_cp, average='macro')) 

rm_svm = len(veracity_pred) - len(veracity_true_cp)

veracity_true_cp = deepcopy(veracity_true)
veracity_pred_cp = deepcopy(veracity_pred)

for i,item in reversed(list(enumerate(veracity_true))):
    if all_pred_RF[i]==0:
        del (veracity_true_cp[i])
        del (veracity_pred_cp[i])
        
print (len(veracity_true_cp))
print (len(veracity_pred_cp))

print("Veracity Accuracy after RF rejection",accuracy_score(veracity_true_cp, veracity_pred_cp))
print("Veracity Macro F after RF rejection",f1_score(veracity_true_cp, veracity_pred_cp, average='macro')) 

rm_rf = len(veracity_pred) - len(veracity_true_cp)

#%%
utype = 'variation_ratio'
print (utype)
num_inst = len(veracity_true)

#int(num_inst*(1-0.975)),int(num_inst*(1-0.95)),int(num_inst*(1-0.9)),
#                    int(num_inst*(1-0.85)),int(num_inst*(1-0.8)),int(num_inst*(1-0.7)),
#                    int(num_inst*(1-0.6)),int(num_inst*(1-0.5)),

del_outlier_list = [ rm_svm, rm_rf]

if utype=='aleatoric':
    uncertainty=aleatoric_uncertainty
elif utype=='entropy':
    uncertainty=predictive_entropy
elif utype=='variation_ratio':
    uncertainty=variation_ratio
elif utype=='variance':
    uncertainty=variance
elif utype=='softmax':
    uncertainty=softmax
    
#uncertainty_cp = deepcopy(uncertainty)

for del_outlier in del_outlier_list:
    
    uncertainty_cp = deepcopy(uncertainty)
    veracity_true_cp = deepcopy(veracity_true)
    veracity_pred_cp = deepcopy(veracity_pred)
    #RAND
    #for _ in range(del_outlier):
    ##    print ("removing: ", true[np.argmax(uncertainty)],predicted[np.argmax(uncertainty)],uncertainty[np.argmax(uncertainty)])
    #    randind = random.randint(0,len(true)) 
    #    del true[randind]
    #    del predicted[randind]
    #    del uncertainty[randind]
    #MAX
    if utype!='softmax':
        for _ in range(del_outlier):
        #    print ("removing: ", true[np.argmax(uncertainty)],predicted[np.argmax(uncertainty)],uncertainty[np.argmax(uncertainty)])
            del veracity_true_cp[np.argmax(uncertainty_cp)]
            del veracity_pred_cp[np.argmax(uncertainty_cp)]
            del uncertainty_cp[np.argmax(uncertainty_cp)]
    #MIN
    else:
        for _ in range(del_outlier):
#            print ("removing: ", true[np.argmin(uncertainty)],predicted[np.argmin(uncertainty)],uncertainty[np.argmin(uncertainty)])
            del veracity_true_cp[np.argmin(uncertainty_cp)]
            del veracity_pred_cp[np.argmin(uncertainty_cp)]
            del uncertainty_cp[np.argmin(uncertainty_cp)]

    print(del_outlier)
    print( np.round(accuracy_score(veracity_true_cp, veracity_pred_cp),3))
    print( np.round(f1_score(veracity_true_cp, veracity_pred_cp, average='macro'),3))




    
