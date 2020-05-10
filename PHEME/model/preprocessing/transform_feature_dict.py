import numpy as np
from tree2branches import tree2branches, tree2timeline
#%%
def transform_feature_dict(thread_feature_dict, conversation, sequential = True, feature_set = ['avgw2v'], timeline=False  ):

    thread_features_array = []
    
    if sequential:
       if timeline:
           branches = tree2timeline(conversation)
           
       else:
           branches = tree2branches(conversation['structure']) # the problem is that there are branches that include tweets that are not among the replies
       for branch in branches:
           branch_rep = []
           for twid in branch:
               if twid in thread_feature_dict.keys():
                   
                   tweet_rep = dict_to_array(thread_feature_dict[twid],feature_set)
                   branch_rep.append(tweet_rep)
               
           if branch_rep!=[]:   
              
               branch_rep = np.asarray(branch_rep)
               thread_features_array.append(branch_rep)    
       thread_features_array = np.asarray(thread_features_array)  
       
    else:
        
       thread_features_array = dict_to_array(thread_feature_dict[conversation['id']],feature_set =feature_set )
    
    return thread_features_array

#%%

def dict_to_array(feature_dict, feature_set = ['avgw2v']):
    tweet_rep = []
    for feature_name in feature_set:
        if feature_name in list(feature_dict.keys()):
            if np.isscalar(feature_dict[feature_name]) :
                tweet_rep.append(feature_dict[feature_name])
            else:
                tweet_rep.extend(feature_dict[feature_name])
                
        else:
            print (feature_name)
            
    tweet_rep =  np.asarray(tweet_rep)
    return tweet_rep


