from keras.models import Model
from keras.layers import Input,LSTM, Dense, Masking, Dropout, TimeDistributed, Lambda, Activation 
from keras import regularizers
from keras import optimizers
from keras import backend as K
from tensorflow.contrib import distributions
import numpy as np
from keras import regularizers
from keras import optimizers
from keras.layers.merge import concatenate
from sklearn.metrics import f1_score, accuracy_score
#from branch2treelabels import branch2treelabels
from numpy.random import seed
from tensorflow import set_random_seed
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import pickle
#import models
#from evaluation import eval_timeline, eval_branches
#from convert2trees import branch2tree
#from parameter_search import parameter_search, objective_function_branch, objective_function_timeline
#%%
class KerasDropoutPrediction(object):
        def __init__(self,model):
            self.f = K.function(
                    [model.layers[0].input, 
                     K.learning_phase()],
                    [model.layers[-1].output])
        def predict(self,x, n_iter=10):
            result = []
            
            for _ in range(n_iter):
                result.append(np.squeeze(np.array(self.f([x , 1]))))
                
            result = np.array(result)
    #        .reshape(n_iter,len(x)).T
            
            return result
        
def predictive_entropy(prob):
    return -np.sum(np.log(prob) * prob) 
#%%
def load_model(foldid, dp):
    class KerasDropoutPrediction(object):
        def __init__(self,model):
            self.f = K.function(
                    [model.layers[0].input, 
                     K.learning_phase()],
                    [model.layers[-1].output])
        def predict(self,x, n_iter=10):
            result = []
            
            for _ in range(n_iter):
                result.append(np.squeeze(np.array(self.f([x , 1]))))
                
            result = np.array(result)
    #        .reshape(n_iter,len(x)).T
            
            return result
    def predictive_entropy(prob):
    	return -np.sum(np.log(prob) * prob)    
    #def montecarlo_prediction(model, X_data, T):
    #	# shape: (T, N, C)
    #    K.learning_phase(1)
    #    predictions = np.array([model.predict(X_data) for _ in range(T)])
    #
    #	# shape: (N, C)
    #    prediction_means = np.mean(predictions, axis=0)
    #	
    #	# shape: (N)
    #    prediction_variances = np.apply_along_axis(predictive_entropy, axis=1, arr=prediction_means)
    #    return (prediction_means, prediction_variances)
    #%%
    # standard categorical cross entropy
    # N data points, C classes
    # true - true values. Shape: (N, C)
    # pred - predicted values. Shape: (N, C)
    # returns - loss (N)
    def categorical_cross_entropy(true, pred):
    	return np.sum(true * np.log(pred), axis=1)
    
    # Bayesian categorical cross entropy.
    # N data points, C classes, T monte carlo simulations
    # true - true values. Shape: (N, C)
    # pred_var - predicted logit values and variance. Shape: (N, C + 1)
    # returns - loss (N,)
    def bayesian_categorical_crossentropy(T, num_classes):
      def bayesian_categorical_crossentropy_internal(true, pred_var):
        # shape: (N,)
        std = K.sqrt(pred_var[:, num_classes:])
        # shape: (N,)
        #variance = pred_var[:, num_classes]
        #variance_depressor = K.exp(variance) - K.ones_like(variance)
        # shape: (N, C)
        pred = pred_var[:, 0:num_classes]
        # shape: (N,)
        #undistorted_loss = K.categorical_crossentropy(true, pred, from_logits=True)
        # shape: (T,)
        iterable = K.variable(np.ones(T))
        dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
        
        monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, pred, dist, num_classes), iterable, name='monte_carlo_results')
        
        variance_loss = K.mean(monte_carlo_results, axis=0) # * undistorted_loss
        
        return  variance_loss #+ undistorted_loss # + variance_depressor
      
      return bayesian_categorical_crossentropy_internal
    
    # for a single monte carlo simulation, 
    #   calculate categorical_crossentropy of 
    #   predicted logit values plus gaussian 
    #   noise vs true values.
    # true - true values. Shape: (N, C)
    # pred - predicted logit values. Shape: (N, C)
    # dist - normal distribution to sample from. Shape: (N, C)
    # undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
    # num_classes - the number of classes. C
    # returns - total differences for all classes (N,)
    def gaussian_categorical_crossentropy(true, pred, dist, num_classes):
      def map_fn(i):
        std_samples = K.transpose(dist.sample(num_classes))
        distorted_loss = K.categorical_crossentropy(true, pred + std_samples,from_logits=True)
        #diff = undistorted_loss - distorted_loss
        return distorted_loss #-K.elu(diff)
      return map_fn
    #%%
    def heteroscedastic_model(x_train, y_train, params, output_classes=3):
        
        num_lstm_units = int(params['num_lstm_units'])
        num_dense_units = int(params['num_dense_units'])
        
        num_dense_layers = int(params['num_dense_layers'])
        num_lstm_layers = int(params['num_lstm_layers'])
        
#        num_epochs = params['num_epochs']
        learn_rate = params['learn_rate']
#        mb_size = params['mb_size']
#        l2reg = params['l2reg']
        drop_rate = params['dropout']
        rng_seed = params['rng_seed']
        
        seed(rng_seed)
        set_random_seed(rng_seed)
        
        num_features = x_train.shape[2]
        
        inputs = Input(shape=(None,num_features), name="input")
         
        mask = Masking(mask_value=0.)(inputs)
        if num_lstm_layers==1:
            lstm1 = LSTM(num_lstm_units, return_sequences=False, dropout = drop_rate, recurrent_dropout = drop_rate)(mask)
        elif num_lstm_layers==2 :
            lstm1 = LSTM(num_lstm_units, return_sequences=True, dropout = drop_rate, recurrent_dropout = drop_rate)(mask)
            lstm2 = LSTM(num_lstm_units, return_sequences=False, dropout = drop_rate, recurrent_dropout = drop_rate)(lstm1)
            lstm1=lstm2
        elif num_lstm_layers>2:
            lstm1 = LSTM(num_lstm_units, return_sequences=True, dropout = drop_rate, recurrent_dropout = drop_rate)(mask)
            for i in range(num_lstm_layers-2):
                lstm2 = LSTM(num_lstm_units, return_sequences=False, dropout = drop_rate, recurrent_dropout = drop_rate)(lstm1)
                lstm1=lstm2
            lstm2 = LSTM(num_lstm_units, return_sequences=False, dropout = drop_rate, recurrent_dropout = drop_rate)(lstm1)
            lstm1=lstm2
            
        hidden1 = Dense(num_dense_units)(lstm1)
        x = Dropout(drop_rate)(hidden1)
        
        for i in range(num_dense_layers-1):
             hidden2 = Dense(num_dense_units)(x)
             x = Dropout(drop_rate)(hidden2)
            
        logits = Dense(output_classes)(x)
        variance_pre = Dense(1)(x)
        variance = Activation('softplus', name='variance')(variance_pre)
        logits_variance = concatenate([logits, variance], name='logits_variance')
        softmax_output = Activation('softmax', name='softmax_output')(logits)
    
        model = Model(inputs=inputs, outputs=[logits_variance,softmax_output])
        
        model.compile(
        optimizer=optimizers.Adam(lr=learn_rate),
        loss={'logits_variance': bayesian_categorical_crossentropy(T = 10, num_classes=3),
              'softmax_output': 'categorical_crossentropy'},
        metrics={'softmax_output': 'categorical_accuracy'},
        loss_weights={'logits_variance': .2, 'softmax_output': 1.})
        
    #    model.fit(x_train, [y_train, y_train], batch_size=mb_size, epochs=num_epochs, shuffle=False, class_weight=None)
    
        return model
    #%%
    #%%
    
    # branch code
    
    path = "saved_data_fullPHEME/ebola-essien"
    
    x_train = np.load(os.path.join(path, 'train_array.npy'))
    y_train = np.load(os.path.join(path, 'labels.npy'))
#    ids_train = np.load(os.path.join(path, 'train/ids.npy'))
    
#    x_dev = np.load(os.path.join(path, 'dev/train_array.npy'))
#    y_dev = np.load(os.path.join(path, 'dev/labels.npy'))
#    ids_dev = np.load(os.path.join(path, 'dev/ids.npy'))
    
#    x_test = np.load(os.path.join(path, 'test/train_array.npy'))
#    y_test = np.load(os.path.join(path, 'test/labels.npy'))
#    ids_test = np.load(os.path.join(path, 'test/ids.npy'))
    
    # join dev and train
    
#    x_dev = pad_sequences(x_dev, maxlen=len(x_train[0]), dtype='float32', padding='post', truncating='post', value=0.)
#    x_train = np.concatenate((x_train, x_dev), axis=0)
#    ids_train = np.concatenate((ids_train, ids_dev), axis=0)
#    y_train = np.concatenate((y_train, y_dev), axis=0)
    y_train = to_categorical(y_train, num_classes=None)
#    y_test = to_categorical(y_test, num_classes=None)
    
    params_file = "output_dp01_main/"
    filepath_branch_avgw2v_aleatoric_as_papers = "output_dp01_main/"
#     filepath_branch_avgw2v_aleatoric_as_papers = ""
    
#    with open(params_file+"bestparams_avgw2v.txt", 'rb') as f:
#        params = pickle.load(f)
    
    with open(params_file+"bestparams_avgw2v.txt", 'rb') as f:
        params = pickle.load(f)
    params['dropout']=dp
    model = heteroscedastic_model(x_train, y_train, params, output_classes=3)
    
    model_filename = filepath_branch_avgw2v_aleatoric_as_papers+"model"+foldid+".h5"
    
    model.load_weights(model_filename)

    return model, params
#%%

def predict_on_data(model,params, x_test, y_test, num_classes=3, verbose=False):
    
#%%
    mb_size = params['mb_size']
    
#    predictions_train = model.predict(x_train, batch_size=mb_size, verbose=verbose)
    predictions_test = model.predict(x_test, batch_size=mb_size, verbose=verbose)
    
#    aleatoric_uncertainties_train = np.reshape(predictions_train[0][:,num_classes:], (-1))
    aleatoric_uncertainties_test = np.reshape(predictions_test[0][:,num_classes:], (-1))
    
#    logits_train = predictions_train[0][:,0:num_classes]
    logits_test = predictions_test[0][:,0:num_classes]
	
	# Shape (N, C)
#    softmax_train = predictions_train[1]
    softmax_test = predictions_test[1]

#    p_train = np.argmax(softmax_train, axis=1)
    p_test = np.argmax(softmax_test, axis=1)
    
    n_iter = 100
    kdp = KerasDropoutPrediction(model)
    y_pred_do_test = kdp.predict(x_test,n_iter=n_iter)
    prediction_means_test = np.mean(y_pred_do_test, axis=0)
    prediction_var_test = np.var(y_pred_do_test, axis=0)
    if x_test.shape[0]>1:
        prediction_var_max_test = np.max(prediction_var_test, axis=1)
        predictive_entropy_test = np.apply_along_axis(predictive_entropy, axis=1, arr=prediction_means_test)
        predictions_test = np.argmax(y_pred_do_test, axis=2)
        temp1 = [np.unique(predictions_test[:,i], return_counts=True) for i in range(len(y_test))]
        temp2 = [temp1[i][1] for i in range(len(y_test))]
        temp3 = [np.max(temp2[i]) for i in range(len(y_test))]
        variation_ratio_test = [(n_iter-temp3[i])/n_iter for i in range(len(y_test))]
        prediction_from_avg_softmax = np.argmax(prediction_means_test, axis=1)
    elif  x_test.shape[0]==1:
        prediction_var_max_test = prediction_var_test
        predictive_entropy_test = predictive_entropy(prediction_means_test)
        predictions_test = np.argmax(y_pred_do_test, axis=1)
        temp1 = np.unique(predictions_test, return_counts=True)
        temp2 = temp1[1]
        temp3 = np.max(temp2) 
        variation_ratio_test =(n_iter-temp3)/n_iter 
        prediction_from_avg_softmax = np.argmax(prediction_means_test)
    
    
    test_results = {'logits_raw': logits_test,
                    'softmax_raw':softmax_test,
		           'prediction_from_softmax_raw':p_test,
		           
		           'true_label':y_test, # np.argmax(y_test, axis=1),
                    'true_label_expanded':y_test,
                    
                    'avg_softmax_from_samples':prediction_means_test,
                    
#                    'softmax_from_samples': y_pred_do_test,
                    
		           'prediction_from_avg_softmax':prediction_from_avg_softmax,
                   
                    'variation_ratio':variation_ratio_test,
                   
		           'aleatoric_uncertainty':aleatoric_uncertainties_test,
		           
                    'variance_3':prediction_var_test,
                    'variance_1':prediction_var_max_test,
                    'predictive_entropy':predictive_entropy_test}

#    result = {'train': train_result, 'test': test_results}        


    return test_results



def get_sorted_replies(folds, foldid, treeid):

    for i,conv in enumerate(folds[foldid]):
        if folds[foldid][i]['id']==treeid:
            raw_tree = folds[foldid][i]
    
    replies = raw_tree['replies']
    replies_timestamp = [reply['created_at'] for reply in replies]
    replies_idstr = [reply['id'] for reply in replies]
    sorted_replies_idstr = [x for (y,x) in sorted(zip(replies_timestamp,replies_idstr))]
    sorted_replies = []
    for reply_id in sorted_replies_idstr:
        for rep in replies:
            if rep['id']==reply_id:
                sorted_replies.append(rep)
    return sorted_replies
















