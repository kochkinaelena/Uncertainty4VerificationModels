import numpy as np
np.random.seed(123)

#from LSTM_model import LSTM_model, build_LSTM_model
from rmse import rmse
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score,precision_recall_fscore_support
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels
import pickle
import json
import models
from convert2trees import branch2tree
from copy import deepcopy
from evaluation import eval_branches
from models import heteroscedastic_model


def eval_veracity_LSTM_CV(params,dataset='15'):   

    path = 'preprocessing/saved_data_'+dataset
    
    folds = ['0','1', '2','3', '4']

    num_epochs = params['num_epochs'] 
    mb_size = params['mb_size']
    

    for f in folds:
        
        print(f)
        test = f+'/test'
        train = f+'/train'
        
        x_test = np.load(os.path.join(path,test, 'train_array.npy'))
        y_test = np.load(os.path.join(path,test, 'labels.npy'))
        ids_test = np.load(os.path.join(path,test, 'ids.npy'))
        
        predictions_train = []

        
        x_train = np.load(os.path.join(path,train, 'train_array.npy'))
        y_train = np.load(os.path.join(path,train, 'labels.npy'))
        y_train = to_categorical(y_train, num_classes=4)
        ids_train = np.load(os.path.join(path,train, 'ids.npy'))
        
        model = heteroscedastic_model(x_train, y_train, params, output_classes=4)
        model.fit(x_train,[y_train, y_train], batch_size=mb_size,
                  epochs=num_epochs, shuffle=False, class_weight=None)
        
        predictions = models.predict_on_data(model,params, x_train, y_train, x_test, y_test, num_classes=3, verbose=True)
        
        tree_results_train = branch2tree(ids_train, predictions['train'])
        
        predictions['train']['tree_results'] = tree_results_train
        
        predictions_train.append(predictions['train'])

        filename = 'output/model'+f+'.h5'
        model.save(filename)
        json_string = model.to_json()
        with open('output/my_model_architecture'+f+'.h5','w') as fout:
            json.dump(json_string,fout)
        
        model.save_weights('output/my_model_weights'+f+'.h5')
        # I need to improve this
        
        
        tree_results_test = branch2tree(ids_test, predictions['test'])
        
        
        predictions['test']['tree_results'] = tree_results_test
        predictions['train']['tree_results'] = predictions_train
        
        
        filename = 'output/predictions'+f+'.pkl'
        fout = open(filename, "wb")
        pickle.dump(predictions, fout)
        fout.close()
        
        eval_info_test = eval_branches(tree_results_test)
#        eval_info_train = [eval_branches(i)  for i in tree_results_train]
        
        eval_info = {}
        eval_info['test'] = eval_info_test
#        eval_info['train'] = eval_info_train
        
        filename = 'output/eval_info'+f+'.pkl'
        fout = open(filename, "wb")
        pickle.dump(eval_info, fout)
        fout.close()

