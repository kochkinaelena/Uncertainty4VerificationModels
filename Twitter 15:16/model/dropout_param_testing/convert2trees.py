import numpy as np
from copy import deepcopy
def branch2tree(ids_test, branch_results):
#%%    
    trees = np.unique(ids_test)
    
    result = {}
    #%%
    for tree in trees:
        
        result[tree] = {}
        
        treeindx = np.where(ids_test == tree)[0]
    #%%    
        branch = {}
        result[tree]['branches'] = []
        for i in treeindx:
            for key in list(branch_results.keys()):
                branch[key] = branch_results[key][i]
                
            result[tree]['branches'].append(deepcopy(branch))
            
    #%%        
        list_of_keys = ['softmax_raw', 'avg_softmax_from_samples',
                        'aleatoric_uncertainty', 'predictive_entropy', 
                        'variance_3', 'variance_1', 'variation_ratio']
        for key in list(list_of_keys):
            result[tree][key+'raw_avg'] = [branch_results[key][i] for i in treeindx]
            result[tree][key+'_avg'] = np.mean(result[tree][key+'raw_avg'],axis=0)
        
        result[tree]['true_label'] = [branch_results['true_label'][i] for i in treeindx][0]
    #%%
        
        result[tree]['tree_prediction_from_avg_softmax'] = np.argmax(result[tree]['avg_softmax_from_samples_avg'])
        result[tree]['tree_prediction_from_avg_avg_softmax'] = np.argmax(result[tree]['softmax_raw_avg'])
    #%%
        result[tree]['branch_predictions'] = [branch_results['prediction_from_softmax_raw'][i] for i in treeindx]
        unique, counts = np.unique(result[tree]['branch_predictions'], return_counts=True)
        result[tree]['tree_prediction_majority_vote'] = unique[np.argmax(counts)]
        
    #%%
        
        result[tree]['tree_prediction_weighted_avg_aleatoric'] = [result[tree]['softmax_rawraw_avg'][i]*result[tree]['aleatoric_uncertaintyraw_avg'][i]
                                                        for i in range(len(result[tree]['aleatoric_uncertaintyraw_avg']))]
        result[tree]['tree_prediction_weighted_avg_aleatoric'] = np.argmax(np.mean(result[tree]['tree_prediction_weighted_avg_aleatoric'], axis=0))
        
        result[tree]['tree_prediction_weighted_avg_epistemic'] = [result[tree]['softmax_rawraw_avg'][i]*result[tree]['variance_1raw_avg'][i]
                                                        for i in range(len(result[tree]['variance_1raw_avg']))]
        result[tree]['tree_prediction_weighted_avg_epistemic'] = np.argmax(np.mean(result[tree]['tree_prediction_weighted_avg_epistemic'], axis=0))
     
        result[tree]['tree_prediction_weighted_avg_entropy'] = [result[tree]['softmax_rawraw_avg'][i]*result[tree]['predictive_entropyraw_avg'][i]
                                                        for i in range(len(result[tree]['predictive_entropyraw_avg']))]
        result[tree]['tree_prediction_weighted_avg_entropy'] = np.argmax(np.mean(result[tree]['tree_prediction_weighted_avg_entropy'], axis=0))
        
    #%%    
    return result
