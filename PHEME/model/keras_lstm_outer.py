import numpy as np
np.random.seed(364)
from parameter_search import parameter_search
from evaluation_functions import eval_veracity_LSTM_CV 
from RumEv_objective import objective_function_veracity_branchLSTM_fullPHEME 
import pickle
import models
from prep_pipeline import prep_pipeline
from generate_feature_set_list import generate_feature_set_list    
#%%
ntrials = 1

# Feature search and parameter search for full PHEME
feature_set_list, feature_name_list = generate_feature_set_list()

for i,fs in enumerate(feature_set_list):
    prep_pipeline(dataset='fullPHEME',feature_set = fs)
    params = parameter_search(ntrials, objective_function_veracity_branchLSTM_fullPHEME,  feature_name_list[i])    
    eval_veracity_LSTM_CV(params, feature_name_list[i])
    
#%%















