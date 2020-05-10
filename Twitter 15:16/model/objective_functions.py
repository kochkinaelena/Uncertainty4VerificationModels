from hyperopt import STATUS_OK
from models import heteroscedastic_model
from sklearn.metrics import f1_score
import numpy as np
import os
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels
from keras.preprocessing.sequence import pad_sequences
#%%

def objective_function_branchLSTM_Twitter15(params):
#%%    
    path = 'preprocessing/saved_data_15'
    
#   fold 0 is development set
    
    train = '0/train'
    test = '0/test'
    
    max_branch_len = 25
    
    x_train = []
    y_train = []
    

    temp_x_train = np.load(os.path.join(path,train, 'train_array.npy'))
    y_train = np.load(os.path.join(path,train, 'labels.npy'))
    
    #   pad sequences to the size of the largest 
    x_train = pad_sequences(temp_x_train, maxlen=max_branch_len, dtype='float32', padding='post', truncating='post', value=0.)
        
        
    x_test = np.load(os.path.join(path,test, 'train_array.npy'))
    y_test = np.load(os.path.join(path,test, 'labels.npy'))
    ids_test = np.load(os.path.join(path,test, 'ids.npy'))
    
    #%
    y_train = to_categorical(y_train, num_classes=None)
    
#    y_pred, confidence = LSTM_model(x_train, y_train, x_test, params)
    
    model = heteroscedastic_model(x_train, y_train, params, output_classes=4)
    mb_size = params['mb_size']
    num_epochs = params['num_epochs']
    model.fit(x_train, [y_train, y_train], batch_size=mb_size, epochs=num_epochs, shuffle=False, class_weight=None)

    verbose = False
    predictions_test = model.predict(x_test, batch_size=mb_size, verbose=verbose)
    softmax_test = predictions_test[1]
    y_pred = np.argmax(softmax_test, axis=1)
    
    trees, tree_prediction, tree_label = branch2treelabels(ids_test,y_test,y_pred)
    
    
    mactest_F = f1_score(tree_label, tree_prediction, average='macro')
    
    output = {'loss': 1-mactest_F,
              'Params': params,
              'status': STATUS_OK}
#%%    
    return output
#out = objective_function_branchLSTM_Twitter15(params)
    

def objective_function_branchLSTM_Twitter16(params):
#%%    
    path = 'preprocessing/saved_data_16'
    
#   fold 0 is development set
    
    train = '0/train'
    test = '0/test'
    
    max_branch_len = 25
    
    x_train = []
    y_train = []
    

    temp_x_train = np.load(os.path.join(path,train, 'train_array.npy'))
    y_train = np.load(os.path.join(path,train, 'labels.npy'))
    
    #   pad sequences to the size of the largest 
    x_train = pad_sequences(temp_x_train, maxlen=max_branch_len, dtype='float32', padding='post', truncating='post', value=0.)
        
        
    x_test = np.load(os.path.join(path,test, 'train_array.npy'))
    y_test = np.load(os.path.join(path,test, 'labels.npy'))
    ids_test = np.load(os.path.join(path,test, 'ids.npy'))
    
    #%
    y_train = to_categorical(y_train, num_classes=None)
        
    model = heteroscedastic_model(x_train, y_train, params, output_classes=4)
    mb_size = params['mb_size']
    num_epochs = params['num_epochs']
    model.fit(x_train, [y_train, y_train], batch_size=mb_size, epochs=num_epochs, shuffle=False, class_weight=None)

    verbose = False
    predictions_test = model.predict(x_test, batch_size=mb_size, verbose=verbose)
    softmax_test = predictions_test[1]
    y_pred = np.argmax(softmax_test, axis=1)
    
    trees, tree_prediction, tree_label = branch2treelabels(ids_test,y_test,y_pred)
    
    
    mactest_F = f1_score(tree_label, tree_prediction, average='macro')
    
    output = {'loss': 1-mactest_F,
              'Params': params,
              'status': STATUS_OK}
#%%    
    return output

