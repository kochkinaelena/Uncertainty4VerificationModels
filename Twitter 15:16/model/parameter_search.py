
import pickle
from hyperopt import fmin, tpe, hp, Trials 


def parameter_search(ntrials, objective_function, fname):

    search_space = {'num_dense_layers': hp.choice('nlayers', [1, 2]),
                    'num_dense_units': hp.choice('num_dense', [200, 300,
                                                               400, 500]),
                    'num_epochs': hp.choice('num_epochs',  [30, 50, 100, 200]),
                    'num_lstm_units': hp.choice('num_lstm_units', [100, 200,
                                                                   300]),
                    'num_lstm_layers': hp.choice('num_lstm_layers', [1, 2]),
                    'learn_rate': hp.choice('learn_rate', [1e-4, 3e-4, 1e-3]),
                    'mb_size': hp.choice('mb_size', [32, 64]),
                    'l2reg': hp.choice('l2reg', [ 1e-4, 3e-4, 1e-3]),
                    'rng_seed': hp.choice('rng_seed', [364]),
                    'dropout': hp.choice('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
                    }
    
    trials = Trials()
    
    best = fmin(objective_function,
        space=search_space,
        algo=tpe.suggest,
        max_evals=ntrials,
        trials=trials)
    
#    print best
    
    bp = trials.best_trial['result']['Params']
    
    f = open('trials_'+fname+'.txt', "wb")
    pickle.dump(trials, f)
    f.close()
    
    filename = 'bestparams_'+fname+'.txt'
    f = open(filename, "wb")
    pickle.dump(bp, f)
    f.close()
    
    return bp