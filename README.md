# Estimating predictive uncertainty for rumour verification models
*Code for the experiments in the ACL 2020 paper "Estimating predictive uncertainty for rumour verification models"*

*Documentation and more analysis code is being added*

## Description of contents

This repo contains 2 sub-folders: `PHEME` and `Twitter 15/16`. 

The directory structure and the code in both directories is analogous, except for some difference in preprocessing as the datasets come in different formats. Thus, `Twitter 15/16` directory includes code to download tweets using their ids in `download_tweets`. 

For both datasets `model` directories for both datasets contain the code to preprocess the data in `preprocessing`, which should be run first and it will generate `saved_data` folders that will be used as input to the models. 

The `models.py` file contains model definition. 

The `keras_lstm_outer.py` is the outer most file that pulls in functions from other files in the folder to allow to run hyper-parameter optimisation using `parameter_search.py` and `objective_functions.py`. After parametes are selected `keras_lstm_outer.py` will call evaluation function from `evaluation_functions.py` that will generate output. The produced output will be used for analysis. 

Code to perform analysis of runs, such as supervised and unsupervised rejection is stored in `analysis` folder. 

## Requirements

### Datasets

**PHEME**

https://figshare.com/articles/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078

**Twitter 15 and Twitter 16**

https://github.com/majingCUHK/Rumor_RvNN
https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0


### Python packages
`numpy, sklearn, scipy, pickle, matplotlib, nltk, gensim, hyperopt, keras, tensorflow`


## How to run

1. Download the dataset(s) above

2. Set filepaths 

3. Run pre-processing

`model/preprocessing/prep_pipeline.py`

4. Run the model (with or without hyper-parameter optimisation)

Change `model/keras_lstm_outer.py` to either **(a)** load a set of hyper-parameters from existing `bestparams.txt` file or **(b)** to run  hyper-parameter optimisation using `parameter_search` function, set `ntrials` variable, a number of parameter combinations to be evaluated.

Then run:

`python model/keras_lstm_outer.py`

5. Run analysis on the output produced by the model

``


If you have any questions feel free to contact me E.Kochkina@warwick.ac.uk 
