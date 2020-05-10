import os 
import numpy as np
import json
import gensim
import nltk
import re
from nltk.corpus import stopwords
from copy import deepcopy
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
import help_prep_functions
#from lexer import lexicon_reader
#from extract_thread_features import extract_thread_features, extract_thread_features_incl_response #before using these functions use loadW2Vmodel
#from get_bow_dict import get_bow_dict

#%%
# This is reading and preprocessing and saving 

def listdir_nohidden(path_to_rumours):
    folds = os.listdir(path_to_rumours)
    newfolds = [i for i in folds if i[0] != '.']
    
    return newfolds


#%%
def read_fullPHEME( rnrfolds = ['germanwings-crash', 'ferguson','charliehebdo',
                    'ottawashooting','sydneysiege', 'putinmissing',
                    'prince-toronto','ebola-essien', 'gurlitt']):
# TODO:
#    help_prep_functions.loadW2vModel()
    rnr_annotations = []
    path_to_rnr_folds = 'all-rnr-annotated-threads'
    
    conversation = {}
    noreplyconv = 0
#    rnrfolds = ['germanwings-crash-all-rnr-threads', 'ferguson-all-rnr-threads','charliehebdo-all-rnr-threads',
#                    'ottawashooting-all-rnr-threads','sydneysiege-all-rnr-threads']
    
    preprocessed_data = {}
    cvfolds = {}

    for fold in rnrfolds:
        preprocessed_data[fold+'-all-rnr-threads'] = []
        cvfolds[fold] = []
        rnr_annotations = []
        path_to_rumours = os.path.join(path_to_rnr_folds,fold+'-all-rnr-threads','rumours')
        rnrthreads = listdir_nohidden(path_to_rumours)
    
       
        for thread in rnrthreads:
    #            cnt = cnt+1
            path_to_source = os.path.join(path_to_rumours,thread,'source-tweets')
            src_tw_folder = listdir_nohidden(path_to_source)
            
            path_to_source_tw = os.path.join(path_to_source,src_tw_folder[0])
            with open(path_to_source_tw) as f:
                    for line in f:
                        src = json.loads(line)
            #FILTER OUT GERMAN TWEETS: if source tweet lang is eng then keep
            if src['lang']=='en':
                
                path_struct = os.path.join(path_to_rumours,thread,'structure.json')
                with open(path_struct) as f:
                        for line in f:
                            struct = json.loads(line)
                # JUST GETTING RID of conversations which do not have structure file (45 in total)   
                # another option is to make up a structure ourselves
                if struct !=[]:
                    if len(struct)>1:
            #            print "Structure has more than one root", 
                        if thread in struct.keys():
                            new_struct = {}
                            new_struct[thread] = struct[thread]
                            struct = new_struct
                        else:
                            new_struct = {}
                            new_struct[thread] = struct
                            struct = new_struct
                        
                    conversation['structure'] = struct 
                    path_to_rnr_annotation = os.path.join(path_to_rumours,thread,'annotation.json')
                    with open(path_to_rnr_annotation) as f:
                        for line in f:
                            an = json.loads(line)
                            an['id'] = thread
                            rnr_annotations.append(an) 
                    conversation['id'] = thread
                    conversation['veracity'] = help_prep_functions.convert_annotations(an,string = True)
                    conversation ['source'] = src
                    tweets = []
                    path_repl = os.path.join(path_to_rumours,thread,'reactions')
                    files_t = listdir_nohidden(path_repl) 
                    if len(files_t)<1:
        #                print "No replies", thread
                        noreplyconv = noreplyconv+1
                    for repl_file in files_t:
                        with open(os.path.join(path_repl,repl_file)) as f:
                            for line in f:
                                tw = json.loads(line)
                                tw['used'] = 0
        #                        tw['set'] = convset
                                tw['label'] = conversation['veracity']
                                tw['conv_id'] = thread
                                tweets.append(tw) 
                                
                                if tw['text'] is None:
                                    print ("Tweet has no text", tw['id'])
                    conversation['replies'] = tweets
                    cvfolds[fold].append(deepcopy(conversation))
                    
                    
    return cvfolds 

#%%

#cvfolds = read_fullPHEME()









