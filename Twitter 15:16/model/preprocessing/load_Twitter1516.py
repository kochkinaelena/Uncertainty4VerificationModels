import numpy
import os
import json
import pickle
#%%
def listdir_nohidden(path):
    folds = os.listdir(path)
    newfolds = [i for i in folds if i[0] != '.']
    return newfolds
#%%

def read_Twitter15_16(dataset="15", set = 'test'):
# loop over folds
    cvfolds = {}    
#    dataset = "15" # 16
    
    path_to_data  = "/preprocessed_data"
    path_to_folds = "/Rumor_RvNN-master/nfold"
    path_to_label = "Rumor_RvNN-master/resource/Twitter"+dataset+"_label_All.txt" #Twitter15_label_All.txt
    
    #%%
    data_folder = listdir_nohidden(os.path.join(path_to_data, "twitter"+dataset))
     # train
#    save_folder = "saved_data/twitter"+dataset
    #%%
    
    labelDic = {}
    for line in open(path_to_label):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        labelDic[eid] = label.lower()   
        
    for i in range(5):
        cvfolds[i] = []
        foldfile = 'RNN'+set+'Set_Twitter'+dataset+str(i)+'_tree.txt'
        for eid in open(os.path.join(path_to_folds,foldfile)):
            eid = eid.rstrip()
        #    if eid not in list(labelDic.keys()): continue
            conversation = {}
            conversation['id'] = eid
            conversation['label'] = labelDic[eid]
            
            if eid in data_folder:
                path_to_structure = os.path.join(path_to_data, "twitter"+dataset, eid, 'structure.pkl')
                with open(path_to_structure,'rb') as f:
                    struct = pickle.load(f)
                conversation['structure'] = struct
                
#                struct[]
                for l in struct:
                    if l[0]=='ROOT':
                        srcid = l[1]
                        
                path_to_tweets = os.path.join(path_to_data, "twitter"+dataset, eid, 'tweets_folder')
                tweets = listdir_nohidden(path_to_tweets)
                tweet_list = []
                if tweets!=[]:
                    for t in tweets:
                        with open(os.path.join(path_to_tweets,t)) as f:
                            for line in f:
                                tw = json.loads(line)
                        if tw['id_str']==srcid:
                            conversation['source'] = tw
                        else:
                            tweet_list.append(tw)
                            
                
                    conversation['replies'] = tweet_list
                    if ('source' in conversation.keys()) and ('replies' in conversation.keys()) and ('structure' in conversation.keys()):
                        cvfolds[i].append(conversation)

    
    return cvfolds 


#cvfolds_tw15 = read_Twitter15_16(dataset="15", set = 'test')
