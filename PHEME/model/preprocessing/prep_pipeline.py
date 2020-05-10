from read_fullPHEME_data import read_fullPHEME
from transform_feature_dict import transform_feature_dict
from extract_thread_features import extract_thread_features, extract_thread_features_incl_response #before using these functions use loadW2Vmodel
import help_prep_functions
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
#from generate_feature_set_list import generate_feature_set_list
#from keras.utils.np_utils import to_categorical
#%%
def convert_label(label):
    if label=="true":
        return(0)
    elif label=="false":
        return(1)
    elif label=="unverified":
        return(2)
    else:
        print(label)
#%%
def prep_pipeline(dataset='RumEv', seq = True, feature_set = ['avgw2v'],fs_name = "text", timeline=False ):
#%%

    path = 'saved_data_'+dataset+fs_name
    
    folds = {}
   
    folds = read_fullPHEME()
        
    help_prep_functions.loadW2vModel()
    for fold in folds.keys():
        print (fold)
        feature_fold = []
        labels = []
        ids = []
        for conversation in folds[fold]:
            thread_feature_dict = extract_thread_features_incl_response(conversation)
    #             Optionally here can collect feature dict and save it to read it instead of raw data?
            thread_features_array = transform_feature_dict(thread_feature_dict, 
                                                           conversation, sequential = seq, 
                                                           feature_set = feature_set, timeline=timeline)
            
#            print thread_features_array.shape
            if seq:
                feature_fold.extend(thread_features_array)
                for i in range(len(thread_features_array)):
                    labels.append(convert_label(conversation['veracity']))
                    ids.append(conversation['id'])
            else:
                feature_fold.append(thread_features_array)  
                labels.append(convert_label(conversation['veracity']))
                ids.append(conversation['id'])
        
        feature_fold = pad_sequences(feature_fold, maxlen=None, dtype='float32',
                                     padding='post', truncating='post', value=0.) 
        labels  = np.asarray(labels)
        
    #
    #        categorical_labels = to_categorical(labels, num_classes=None)
        
        
        path_fold = os.path.join(path,fold)
        if not os.path.exists(path_fold):
            os.makedirs(path_fold)
        np.save(os.path.join(path_fold,'train_array'),feature_fold) 
        np.save(os.path.join(path_fold,'labels'),labels) 
        np.save(os.path.join(path_fold,'ids'),ids) 
        


#%%

prep_pipeline(dataset='fullPHEME', seq = True)


#feature_set_list, feature_name_list = generate_feature_set_list()
#prep_pipeline(dataset='fullPHEME', seq = True, feature_set = feature_set_list[0],fs_name = feature_name_list[0], timeline=False )




