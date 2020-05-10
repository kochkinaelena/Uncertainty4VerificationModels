import numpy as np
np.random.seed(123)
from rmse import rmse
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score,precision_recall_fscore_support
from keras.utils.np_utils import to_categorical
import pickle
import json
import models
from convert2trees import branch2tree
from copy import deepcopy
from evaluation import eval_branches
from models import heteroscedastic_model

def eval_veracity_LSTM_CV(params, branch=True):
#%%    
    if branch:
        path = 'saved_data_fullPHEME'
    else:
        path = 'saved_data_timelinefullPHEME'
    
    
    folds = ['ebola-essien','ferguson', 'gurlitt',
             'ottawashooting', 'prince-toronto', 'putinmissing', 
             'sydneysiege', 'charliehebdo','germanwings-crash' ]

    num_epochs = params['num_epochs'] 
    mb_size = params['mb_size']
    

    for number in range(len(folds)):
        
        x_temp = np.load(os.path.join(path,'ebola-essien', 'train_array.npy'))
        y_temp = np.load(os.path.join(path,'ebola-essien', 'labels.npy'))
        model = heteroscedastic_model(x_temp, y_temp, params, output_classes=3)
#        del x_temp
        
        print(number)
        test = folds[number]
        train = deepcopy(folds)
        del train[number]
        
        x_test = np.load(os.path.join(path,test, 'train_array.npy'))
        y_test = np.load(os.path.join(path,test, 'labels.npy'))
        ids_test = np.load(os.path.join(path,test, 'ids.npy'))
        
        predictions_train = []

        for t in train:
            x_train = np.load(os.path.join(path,t, 'train_array.npy'))
            y_train = np.load(os.path.join(path,t, 'labels.npy'))
            y_train = to_categorical(y_train, num_classes=3)
            ids_train = np.load(os.path.join(path,t, 'ids.npy'))
            
            model.fit(x_train,[y_train, y_train], batch_size=mb_size,
                      epochs=num_epochs, shuffle=False, class_weight=None)
            
            predictions = models.predict_on_data(model,params, x_train, y_train, x_test, y_test, num_classes=3, verbose=True)
            
            tree_results_train = branch2tree(ids_train, predictions['train'])
            
            predictions['train']['tree_results'] = tree_results_train
            
            predictions_train.append(predictions['train'])

        filename = 'output/model'+str(test)+'.h5'
        model.save(filename)
        json_string = model.to_json()
        with open('output/my_model_architecture'+str(test)+'.h5','w') as f:
            json.dump(json_string,f)
        
        model.save_weights('output/my_model_weights'+str(test)+'.h5')
        # I need to improve this
        
        
        tree_results_test = branch2tree(ids_test, predictions['test'])
        
        
        predictions['test']['tree_results'] = tree_results_test
        predictions['train']['tree_results'] = predictions_train
        
        
        filename = 'output/predictions'+str(test)+'.pkl'
        f = open(filename, "wb")
        pickle.dump(predictions, f)
        f.close()
        
        eval_info_test = eval_branches(tree_results_test)
        eval_info = {}
        eval_info['test'] = eval_info_test
        filename = 'output/eval_info'+str(test)+'.pkl'
        f = open(filename, "wb")
        pickle.dump(eval_info, f)
        f.close()

