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
        
        return result
def predictive_entropy(prob):
	return -np.sum(np.log(prob) * prob)    

def categorical_cross_entropy(true, pred):
	return np.sum(true * np.log(pred), axis=1)

def bayesian_categorical_crossentropy(T, num_classes):
  def bayesian_categorical_crossentropy_internal(true, pred_var):
    std = K.sqrt(pred_var[:, num_classes:])
    pred = pred_var[:, 0:num_classes]
    iterable = K.variable(np.ones(T))
    dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
    
    monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, pred, dist, num_classes), iterable, name='monte_carlo_results')
    
    variance_loss = K.mean(monte_carlo_results, axis=0) 
    
    return  variance_loss 
  
  return bayesian_categorical_crossentropy_internal

def gaussian_categorical_crossentropy(true, pred, dist, num_classes):
  def map_fn(i):
    std_samples = K.transpose(dist.sample(num_classes))
    distorted_loss = K.categorical_crossentropy(true, pred + std_samples,from_logits=True)
    return distorted_loss 
  return map_fn
#%%
def heteroscedastic_model(x_train, y_train, params, output_classes=4):
    
    num_lstm_units = int(params['num_lstm_units'])
    num_dense_units = int(params['num_dense_units'])
    
    num_dense_layers = int(params['num_dense_layers'])
    num_lstm_layers = int(params['num_lstm_layers'])
    
    num_epochs = params['num_epochs']
    learn_rate = params['learn_rate']
    mb_size = params['mb_size']
    l2reg = params['l2reg']
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
    loss={'logits_variance': bayesian_categorical_crossentropy(T = 10, num_classes=4),
          'softmax_output': 'categorical_crossentropy'},
    metrics={'softmax_output': 'categorical_accuracy'},
    loss_weights={'logits_variance': .2, 'softmax_output': 1.})
    
    return model

#%%


def predict_on_data(model,params, x_train, y_train, x_test, y_test, num_classes=4, verbose=False):
    
#%%
    mb_size = params['mb_size']
    
    predictions_train = model.predict(x_train, batch_size=mb_size, verbose=verbose)
    predictions_test = model.predict(x_test, batch_size=mb_size, verbose=verbose)
    
    aleatoric_uncertainties_train = np.reshape(predictions_train[0][:,num_classes:], (-1))
    aleatoric_uncertainties_test = np.reshape(predictions_test[0][:,num_classes:], (-1))
    
    logits_train = predictions_train[0][:,0:num_classes]
    logits_test = predictions_test[0][:,0:num_classes]
	
	# Shape (N, C)
    softmax_train = predictions_train[1]
    softmax_test = predictions_test[1]

    p_train = np.argmax(softmax_train, axis=1)
    p_test = np.argmax(softmax_test, axis=1)
    
    n_iter = 100
    kdp = KerasDropoutPrediction(model)
    y_pred_do_test = kdp.predict(x_test,n_iter=n_iter)
    prediction_means_test = np.mean(y_pred_do_test, axis=0)
    prediction_var_test = np.var(y_pred_do_test, axis=0)
    prediction_var_max_test = np.max(prediction_var_test, axis=1)
    predictive_entropy_test = np.apply_along_axis(predictive_entropy, axis=1, arr=prediction_means_test)
    predictions_test = np.argmax(y_pred_do_test, axis=2)
    temp1 = [np.unique(predictions_test[:,i], return_counts=True) for i in range(len(y_test))]
    temp2 = [temp1[i][1] for i in range(len(y_test))]
    temp3 = [np.max(temp2[i]) for i in range(len(y_test))]
    variation_ratio_test = [(n_iter-temp3[i])/n_iter for i in range(len(y_test))]
#%%    
    kdp = KerasDropoutPrediction(model)
    y_pred_do_train = kdp.predict(x_train,n_iter=100)
    prediction_means_train = np.mean(y_pred_do_train, axis=0)
    
    predictions_train = np.argmax(y_pred_do_train, axis=2)
    
    temp1 = [np.unique(predictions_train[:,i], return_counts=True) for i in range(len(y_train))]
    temp2 = [temp1[i][1] for i in range(len(y_train))]
    temp3 = [np.max(temp2[i]) for i in range(len(y_train))]
    
    variation_ratio_train = [(n_iter-temp3[i])/n_iter for i in range(len(y_train))]
    
    prediction_var_train = np.var(y_pred_do_train, axis=0)
    prediction_var_max_train = np.max(prediction_var_train, axis=1)
    
    
    
    predictive_entropy_train = np.apply_along_axis(predictive_entropy, axis=1, arr=prediction_means_train)


    train_result = {'logits_raw': logits_train,
                    'softmax_raw':softmax_train,
		           'prediction_from_softmax_raw':p_train,
		           
		           'true_label': np.argmax(y_train,axis=1),
#                    'true_label':y_train,
                    
                    'avg_softmax_from_samples':prediction_means_train, 
                    'prediction_from_avg_softmax':np.argmax(prediction_means_train,axis=1),
                    
                    'variation_ratio':variation_ratio_train,
		           
		           'aleatoric_uncertainty':aleatoric_uncertainties_train,
		           
                    'variance_3':prediction_var_train,
                    'variance_1':prediction_var_max_train,
                    'predictive_entropy':predictive_entropy_train}
    
    test_results = {'logits_raw': logits_test,
                    'softmax_raw':softmax_test,
		           'prediction_from_softmax_raw':p_test,
		           
#		           'true_label': np.argmax(y_test, axis=1),
                    'true_label':y_test,
                    
                    'avg_softmax_from_samples':prediction_means_test,
		           'prediction_from_avg_softmax':np.argmax(prediction_means_test, axis=1),
                   
                    'variation_ratio':variation_ratio_test,
                   
		           'aleatoric_uncertainty':aleatoric_uncertainties_test,
		           
                    'variance_3':prediction_var_test,
                    'variance_1':prediction_var_max_test,
                    'predictive_entropy':predictive_entropy_test}

    result = {'train': train_result, 'test': test_results}        


    return result
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    