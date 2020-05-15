#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Adapted from https://github.com/markus93/NN_calibration

''' 

import numpy as np 

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, correct):
    
    filtered_tuples = [x for x in zip(correct, conf) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == 1])  # How many correct labels
        
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        
        avg_conf = sum([x[1] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        
        accuracy = float(correct)/len_bin  # accuracy of BIN
        
        return accuracy, avg_conf, len_bin


def ECE_score(correct, conf, bin_size = 0.1):
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, correct)     
        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece
