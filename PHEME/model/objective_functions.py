from hyperopt import STATUS_OK
#from LSTM_model import LSTM_model
from models import heteroscedastic_model
from sklearn.metrics import f1_score
import numpy as np
import os
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels
from keras.preprocessing.sequence import pad_sequences


#%%
def objective_function_veracity_branchLSTM_fullPHEME(params):
    path = 'saved_data_fullPHEME'
    
    train = ['ebola-essien','ferguson', 'gurlitt','ottawashooting', 
             'prince-toronto', 'putinmissing', 'sydneysiege']
    
    test = 'charliehebdo'
    max_branch_len = 25
    x_train = []
    y_train = []
    
    for t in train:
        temp_x_train = np.load(os.path.join(path,t, 'train_array.npy'))
        temp_y_train = np.load(os.path.join(path,t, 'labels.npy'))
        
        temp_x_train = pad_sequences(temp_x_train, maxlen=max_branch_len, dtype='float32', padding='post', truncating='post', value=0.)
        
        x_train.extend(temp_x_train)
        y_train.extend(temp_y_train)
    
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
        
    x_test = np.load(os.path.join(path,test, 'train_array.npy'))
    y_test = np.load(os.path.join(path,test, 'labels.npy'))
    ids_test = np.load(os.path.join(path,test, 'ids.npy'))
    #%
    y_train = to_categorical(y_train, num_classes=None)
    
    model = heteroscedastic_model(x_train, y_train, params, output_classes=3)
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

