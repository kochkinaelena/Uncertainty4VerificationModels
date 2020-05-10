from copy import deepcopy

#%%
def add_partial_folds(folds):
#%%

    new_folds = {}
    
    for fold in list(folds.keys()):
        new_folds[fold] = {}
        for tree in folds[fold]:
            id = tree['id']
            new_folds[fold][id] = {}
            replies = tree['replies']
            replies_timestamp = [reply['created_at'] for reply in replies]
            replies_idstr = [reply['id'] for reply in replies]
            
            sorted_replies_idstr = [x for (y,x) in sorted(zip(replies_timestamp,replies_idstr))]
            
            sorted_replies = []
            
            for reply_id in sorted_replies_idstr:
                for rep in replies:
                    if rep['id']==reply_id:
                        sorted_replies.append(rep)
                
            
            for i in range(len(sorted_replies)+1):
                
                new_sorted_replies = deepcopy(sorted_replies[0:i])
#                print (len(new_sorted_replies))
                new_tree = deepcopy(tree)
                new_tree['replies'] = deepcopy(new_sorted_replies)
#                print (len(new_tree['replies']))
                new_folds[fold][id][str(i)] = deepcopy(new_tree)

#%%    
    
    
    return new_folds
    
