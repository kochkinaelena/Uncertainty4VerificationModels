
import numpy as np
np.random.seed(364)
from parameter_search import parameter_search
from evaluation_functions import eval_veracity_LSTM_CV 
from objective_functions import  objective_function_branchLSTM_Twitter15, objective_function_branchLSTM_Twitter16
import pickle
import models

##with open('bestparams.txt', 'rb') as f:
##    params = pickle.load(f)       

#%%
ntrials = 1

params = parameter_search(ntrials, objective_function_branchLSTM_Twitter15, 'avgw2v' )    
eval_veracity_LSTM_CV(params,dataset='15')
    

#params = parameter_search(ntrials, objective_function_branchLSTM_Twitter16, 'avgw2v' )    
#eval_veracity_LSTM_CV(params,dataset='16')








