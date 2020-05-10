import numpy as np
np.random.seed(364)
from loading_model import load_model, predict_on_data
import os
from keras.layers import Dropout
from convert2trees import branch2tree
import pickle
import time
start_time = time.time()
#%%
path = 'saved_data_16'

folds = ['0','1','2','3','4']
#all_pred = []
dropout_values = [0.1, 0.3, 0.5, 0.7] #
for dp in dropout_values:
    for foldid in folds:
        
        model, params = load_model(foldid,dp)
        
        for layer in model.layers:
            if isinstance(layer, Dropout):
    #            layer.rate = dp
                print (layer.get_config())
        
        x_test = np.load(os.path.join(path,foldid, 'test/train_array.npy'))
        y_test = np.load(os.path.join(path,foldid, 'test/labels.npy'))
        ids_test = np.load(os.path.join(path,foldid, 'test/ids.npy'))
    
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

