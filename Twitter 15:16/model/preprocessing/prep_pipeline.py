from load_Twitter1516 import read_Twitter15_16
from transform_feature_dict import transform_feature_dict
from extract_thread_features import extract_thread_features, extract_thread_features_incl_response #before using these functions use loadW2Vmodel
import help_prep_functions
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences

#%%

def convert_label(label):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if label in labelset_nonR:
       y_train = 0#[1,0,0,0]
    if label in labelset_f:
       y_train = 1#[0,1,0,0] 
    if label in labelset_t:
       y_train = 2#[0,0,1,0] 
    if label in labelset_u:
       y_train = 3#[0,0,0,1] 
    return y_train
#%%
def prep_pipeline(dataset='15', seq = True, feature_set = ['avgw2v'], timeline=False ):
#%%

    path = 'saved_data_'+dataset
    
#    if not os.path.exists(path):
#        os.makedirs(path)
#    folds = {}
    if dataset == '15':
        folds_test = read_Twitter15_16(dataset="15", set = 'test')
        folds_train = read_Twitter15_16(dataset="15", set = 'train')
        
    else:
        folds_test = read_Twitter15_16(dataset="16", set = 'test')
        folds_train = read_Twitter15_16(dataset="16", set = 'train')
        
    help_prep_functions.loadW2vModel()
    
    for fold in folds_test.keys():
        
        print (fold)
        
        feature_fold = []
        labels = []
        ids = []
        for conversation in folds_test[fold]:
            #print (conversation['id'])
            
            
            thread_feature_dict = extract_thread_features_incl_response(conversation)
    #             Optionally here can collect feature dict and save it to read it instead of raw data?
            thread_features_array = transform_feature_dict(thread_feature_dict, 
                                                           conversation, sequential = seq, 
                                                           feature_set = feature_set, timeline=timeline)
#            print thread_features_array.shape
            if seq:
                feature_fold.extend(thread_features_array)
                for i in range(len(thread_features_array)):
                    labels.append(convert_label(conversation['label']))
                    ids.append(conversation['id'])
            else:
                feature_fold.append(thread_features_array)  
                labels.append(convert_label(conversation['label']))
                ids.append(conversation['id'])
        
        feature_fold = pad_sequences(feature_fold, maxlen=None, dtype='float32',
                                     padding='post', truncating='post', value=0.) 
        labels  = np.asarray(labels)
        
    #
    #        categorical_labels = to_categorical(labels, num_classes=None)
        
        
        path_fold = os.path.join(path,str(fold), 'test')
        if not os.path.exists(path_fold):
            os.makedirs(path_fold)
#        print feature_fold.shape
        np.save(os.path.join(path_fold,'train_array'),feature_fold) 
        np.save(os.path.join(path_fold,'labels'),labels) 
        np.save(os.path.join(path_fold,'ids'),ids) 
        
    for fold in folds_train.keys():
        
        print (fold)
        
        feature_fold = []
        labels = []
        ids = []
        for conversation in folds_train[fold]:
            
            thread_feature_dict = extract_thread_features_incl_response(conversation)
    #             Optionally here can collect feature dict and save it to read it instead of raw data?
            thread_features_array = transform_feature_dict(thread_feature_dict, 
                                                           conversation, sequential = seq, 
                                                           feature_set = feature_set, timeline=timeline)
#            print thread_features_array.shape
            if seq:
                feature_fold.extend(thread_features_array)
                for i in range(len(thread_features_array)):
                    labels.append(convert_label(conversation['label']))
                    ids.append(conversation['id'])
            else:
                feature_fold.append(thread_features_array)  
                labels.append(convert_label(conversation['label']))
                ids.append(conversation['id'])
        
        feature_fold = pad_sequences(feature_fold, maxlen=None, dtype='float32',
                                     padding='post', truncating='post', value=0.) 
        labels  = np.asarray(labels)
        
    #
    #        categorical_labels = to_categorical(labels, num_classes=None)
        
        
        path_fold = os.path.join(path,str(fold), 'train')
        if not os.path.exists(path_fold):
            os.makedirs(path_fold)
#        print feature_fold.shape
        np.save(os.path.join(path_fold,'train_array'),feature_fold) 
        np.save(os.path.join(path_fold,'labels'),labels) 
        np.save(os.path.join(path_fold,'ids'),ids) 
        
prep_pipeline(dataset='15', seq = True, feature_set = ['avgw2v'], timeline=False )
prep_pipeline(dataset='16', seq = True, feature_set = ['avgw2v'], timeline=False )
