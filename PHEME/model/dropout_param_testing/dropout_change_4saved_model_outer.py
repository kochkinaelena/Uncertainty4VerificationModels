import numpy as np
np.random.seed(364)
from loading_model import load_model, predict_on_data
#from read_fullPHEME_data import read_fullPHEME
import os
from keras.layers import Dropout
from convert2trees import branch2tree
import pickle
import time
start_time = time.time()
#%%
#folds = read_fullPHEME()
path = 'saved_data_fullPHEME'

folds = ['ebola-essien','ferguson', 'gurlitt',
         'ottawashooting', 'prince-toronto', 'putinmissing', 
         'sydneysiege', 'charliehebdo','germanwings-crash' ]
#all_pred = []
dropout_values = [0.1] #, 0.3, 0.5, 0.7
for dp in dropout_values:
    for foldid in folds[0:1]:
        
        model, params = load_model(foldid,dp) # try changing model inside loading function
        
        for layer in model.layers:
            if isinstance(layer, Dropout):
    #            layer.rate = dp
                print (layer.get_config())
        
        x_test = np.load(os.path.join(path,foldid, 'train_array.npy'))
        y_test = np.load(os.path.join(path,foldid, 'labels.npy'))
        ids_test = np.load(os.path.join(path,foldid, 'ids.npy'))
        
        predictions = predict_on_data(model,params, x_test, y_test, num_classes=3, verbose=False)
        
        tree_results_test = branch2tree(ids_test, predictions)
        predictions['tree_results'] = tree_results_test
    
    filepath = 'output/dp'+str(dp)+'/'  
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        
    filename = 'predictions'+foldid+'.pkl'
    
    f = open(filepath+filename, "wb")
    pickle.dump(predictions, f)
    f.close()
        
print("--- %s seconds ---" % (time.time() - start_time))       

 
