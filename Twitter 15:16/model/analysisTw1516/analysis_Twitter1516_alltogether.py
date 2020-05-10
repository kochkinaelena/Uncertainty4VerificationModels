import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score, accuracy_score
import os
from keras.models import load_model
from scipy.stats import mannwhitneyu, kruskal
from copy import deepcopy
import random
random.seed(364)
import pandas as pd
from load_Twitter1516 import read_Twitter15_16
#%%
# load eval and predictions

filepath = ""
folder = "output_T50_weight05"
dataset = "Twitter16"

#%%
# convert predictions 
folds_data = read_Twitter15_16(dataset="16", set = 'test')
#%%
folds = ['0','1','2','3','4']    


true = []
predicted = []
aleatoric_uncertainty = []
predictive_entropy = []
variation_ratio = []
softmax = []
variance = []
tree_n_branches = []
tree_n_tweets = []

for fold in folds:

    whichset = 'test'
#    utype = 'entropy'
#    utype = 'variance'
#    utype = 'softmax'
#    utype = 'aleatoric' # 'entropy'
#    utype = 'variation_ratio'
    del_outlier=0
    nbins = 10
    
    #%%
#    with open(filepath+dataset+"/"+folder+"/output/eval_info"+fold+".pkl", 'rb') as f:
#        evalinfo = pickle.load(f)
        
    with open(filepath+dataset+"/"+folder+"/output/predictions"+fold+".pkl", 'rb') as f:
        predictions = pickle.load(f)
    
    #%%
    prediction = predictions[whichset]['tree_results']

   
    
    for item in list(prediction.keys()):
        
         if  (not np.isnan(prediction[item]['predictive_entropy_avg'])):
             
                 true.append(prediction[item]['true_label'])
                 predicted.append(prediction[item]['tree_prediction_from_avg_avg_softmax']) #tree_prediction_majority_vote
                 
                 aleatoric_uncertainty.append(prediction[item]['aleatoric_uncertainty_avg'])
                 predictive_entropy.append(prediction[item]['predictive_entropy_avg'])
                 softmax.append(np.max(prediction[item]['softmax_raw_avg']))
                 variance.append(prediction[item]['variance_1_avg'])
                 variation_ratio.append(prediction[item]['variation_ratio_avg'])
                 tree_n_branches.append(len(prediction[item]['branches']))
                 
                 thread=None
                 for th in folds_data[int(fold)]:
                     if th['id']==item:
                         thread = th
                         break
                 if thread!=None:
                     tree_n_tweets.append(len(thread['replies'])+1)
                 else:
                     print ("Not found", item)
                     tree_n_tweets.append(0)
                 
                 
                 
         else:
#                 print (item)
             
                 true.append(prediction[item]['true_label'])
                 predicted.append(prediction[item]['tree_prediction_from_avg_avg_softmax']) #tree_prediction_majority_vote
                 
                 aleatoric_uncertainty.append(prediction[item]['aleatoric_uncertainty_avg'])
                 predictive_entropy.append(0)
                 softmax.append(np.max(prediction[item]['softmax_raw_avg']))
                 variance.append(prediction[item]['variance_1_avg'])
                 variation_ratio.append(prediction[item]['variation_ratio_avg'])
                 tree_n_branches.append(len(prediction[item]['branches']))
                 
                 thread=None
                 for th in folds_data[int(fold)]:
                     if th['id']==item:
                         thread = th
                         break
                 if thread!=None:
                     tree_n_tweets.append(len(thread['replies'])+1)
                 else:
                     print ("Not found", item)
                     tree_n_tweets.append(0)

#%%
#
# get highest uncertainty trees

def get_highest_lowest_uncertainty(uncertainty_list, uncertainty_name, prediction, variatio_ratio_treeids=[]):
    n=5

    uncertainty = deepcopy(uncertainty_list)
#    uncertainty.sort()
    
    if variatio_ratio_treeids!=[]:
        uncertainty, variatio_ratio_treeids = zip(*sorted(zip(uncertainty, variatio_ratio_treeids)))
#        uncertainty = [x  for x,_ in sorted(zip(uncertainty,variatio_ratio_treeids))]
#        var_treeid = [x  for _,x in sorted(zip(uncertainty,variatio_ratio_treeids))]
    else:
        uncertainty.sort()

    if uncertainty_name=='avg_softmax_from_samples_avg':
        print ("Lowest "+uncertainty_name+" uncertainty values, conv_id, true, pred")
        for i in uncertainty[0:n]:
            for item in list(prediction.keys()):
                if np.max(prediction[item][uncertainty_name])==i:
                    conv_id = item
                    true = prediction[item]['true_label']
                    pred = prediction[item]['tree_prediction_from_avg_softmax']
                    print (i, conv_id, true, pred)
        print ("Highest "+uncertainty_name+" uncertainty values, conv_id, true, pred")
        for i in uncertainty[-n:]:
            for item in list(prediction.keys()):
                if np.max(prediction[item][uncertainty_name])==i:
                    conv_id = item
                    true = prediction[item]['true_label']
                    pred = prediction[item]['tree_prediction_from_avg_softmax']
                    print (i, conv_id, true, pred)
                    
    elif uncertainty_name=='variation_ratio':
        
        print ("Lowest "+uncertainty_name+" uncertainty values, conv_id, true, pred")
        for ii, i in enumerate(uncertainty[0:n]):
                conv_id = variatio_ratio_treeids[ii]
                true = prediction[conv_id]['true_label']
                pred = prediction[conv_id]['tree_prediction_from_avg_softmax']
                print (i, conv_id, true, pred)
                
        print ("Highest "+uncertainty_name+" uncertainty values, conv_id, true, pred")
        for ii, i in enumerate(uncertainty[-n:]):
                conv_id = variatio_ratio_treeids[len(uncertainty)-n+ii]
                true = prediction[conv_id]['true_label']
                pred = prediction[conv_id]['tree_prediction_from_avg_softmax']
                print (i, conv_id, true, pred)
                    
    else:
    
        print ("Lowest "+uncertainty_name+" uncertainty values, conv_id, true, pred")
        for i in uncertainty[0:n]:
            for item in list(prediction.keys()):
                if prediction[item][uncertainty_name]==i:
                    conv_id = item
                    true = prediction[item]['true_label']
                    pred = prediction[item]['tree_prediction_from_avg_softmax']
                    print (i, conv_id, true, pred)
        print ("Highest "+uncertainty_name+" uncertainty values, conv_id, true, pred")
        for i in uncertainty[-n:]:
            for item in list(prediction.keys()):
                if prediction[item][uncertainty_name]==i:
                    conv_id = item
                    true = prediction[item]['true_label']
                    pred = prediction[item]['tree_prediction_from_avg_softmax']
                    print (i, conv_id, true, pred)
                
    

#get_highest_lowest_uncertainty(aleatoric_uncertainty, 'aleatoric_uncertainty_avg', prediction)
#get_highest_lowest_uncertainty(predictive_entropy, 'predictive_entropy_avg', prediction)
#get_highest_lowest_uncertainty(variance, 'variance_1_avg', prediction)
#get_highest_lowest_uncertainty(softmax, 'avg_softmax_from_samples_avg', prediction)
#get_highest_lowest_uncertainty(variation_ratio, 'variation_ratio', prediction, variatio_ratio_treeids)

#%%

utype_list = ['aleatoric','variation_ratio', 'entropy', 'variance', 'softmax']
pc_list = [1, 0.975,0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5 ]
num_inst = len(true)
del_outlier_list = [0,int(num_inst*(1-0.975)),int(num_inst*(1-0.95)),int(num_inst*(1-0.9)),
                    int(num_inst*(1-0.85)),int(num_inst*(1-0.8)),int(num_inst*(1-0.7)),
                    int(num_inst*(1-0.6)),int(num_inst*(1-0.5))]

#del_outlier_list = [0,18,36,73,110,147,220,294,367]
#del_outlier_list = [34,68,137,206,275,412,550,687]
copy_true = deepcopy(true)
copy_predicted = deepcopy(predicted)

d = {}

for utype in utype_list:
    acc_column = []
    macrof_column = []
    for del_outlier in del_outlier_list:
    
        if utype=='aleatoric':
            uncertainty=deepcopy(aleatoric_uncertainty)
        elif utype=='entropy':
            uncertainty=deepcopy(predictive_entropy)
        elif utype=='variation_ratio':
            uncertainty=deepcopy(variation_ratio)
        elif utype=='variance':
            uncertainty=deepcopy(variance)
        elif utype=='softmax':
            uncertainty=deepcopy(softmax)
            #RAND
        #    for _ in range(del_outlier):
        #    #    print ("removing: ", true[np.argmax(uncertainty)],predicted[np.argmax(uncertainty)],uncertainty[np.argmax(uncertainty)])
        #        randind = random.randint(0,len(true)-1) 
        #        del true[randind]
        #        del predicted[randind]
        #        del uncertainty[randind]
        #MAX
        if utype!='softmax':
            for _ in range(del_outlier):
            #    print ("removing: ", true[np.argmax(uncertainty)],predicted[np.argmax(uncertainty)],uncertainty[np.argmax(uncertainty)])
                del true[np.argmax(uncertainty)]
                del predicted[np.argmax(uncertainty)]
                del uncertainty[np.argmax(uncertainty)]
        #MIN
        if utype=='softmax':
            for _ in range(del_outlier):
        #        print ("removing: ", true[np.argmin(uncertainty)],predicted[np.argmin(uncertainty)],uncertainty[np.argmin(uncertainty)])
                del true[np.argmin(uncertainty)]
                del predicted[np.argmin(uncertainty)]
                del uncertainty[np.argmin(uncertainty)]
        acc_column.append(np.round(accuracy_score(true, predicted),3))
        macrof_column.append(np.round(f1_score(true, predicted, average='macro'),3))
        print (del_outlier)
        print (np.round(accuracy_score(true, predicted),3))
        print (np.round(f1_score(true, predicted, average='macro'),3))
            
        true = deepcopy(copy_true)
        predicted = deepcopy(copy_predicted)
        
    d[utype+'_accuracy'] = acc_column
    d[utype+'_macrof'] = macrof_column
    
d['percentage'] = pc_list
d['num_instances'] = del_outlier_list
df = pd.DataFrame(data=d)
savepath = ""   
df.to_csv(savepath+"rejection_table"+dataset+folder+".csv")
#%%
print (fold)
print (len(softmax))

print( "Accuracy ",np.round(accuracy_score(true, predicted),3))
print( "Macro F ", np.round(f1_score(true, predicted, average='macro'),3))
# Average uncertainty of each type per fold:
print ("softmax ",np.round(np.mean(softmax),3))   
print ("aleatoric_uncertainty ",np.round(np.mean(aleatoric_uncertainty),6))    
print ("variance ",np.round(np.mean(variance),3))       
print ("variation_ratio",np.round(np.mean(variation_ratio),3))  
print ("predictive_entropy ",np.round(np.mean(predictive_entropy),3))      
#%%
fig = plt.figure()  
ax = plt.axes()
title_string = "Histogram of "+utype+" uncertainty"  
ax.set_title(title_string)
ax.hist(uncertainty, bins=nbins)
fig.savefig(title_string+'.png')
#%%

correct = []
incorrect = []
for i in range(len(true)):
    if true[i]==predicted[i]:
        correct.append(uncertainty[i]) 
    else:
        incorrect.append(uncertainty[i]) 

print ("Mean "+utype+" uncertainty - correct group", np.mean(correct), np.median(correct))

print ("Mean "+utype+" uncertainty - incorrect group", np.mean(incorrect), np.median(incorrect))


data = [correct,incorrect]
#fig, ax = plt.subplots()
fig = plt.figure()  
ax = plt.axes()
ax.boxplot(data)
title_string = "Boxplots of correct,incorrect "+utype+" uncertainty" 
ax.set_xticks([1, 2])
ax.set_xticklabels(['Correct', 'Incorrect'])
ax.set_title(title_string)


plt.show()
fig.savefig(title_string+'.png')

#%%


print ("Mann Whitney U tests for "+utype+" uncertainty, correct,incorrect")
print (mannwhitneyu(correct,incorrect))
print ("Kruskal tests for "+utype+" uncertainty, correct,incorrect")
print (kruskal(correct,incorrect))

#%%
#def analyse_class():

utype = "aleatoric"
utype = "epistemic"

#uncertainty = aleatoric_uncertainty
uncertainty = variation_ratio


from matplotlib import rcParams

labelsize = 16
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 

true_lab = []
false_lab = []
unverified_lab = []
nonrum_lab = []

for i in range(len(true)):
    if true[i]==0:
        nonrum_lab.append(uncertainty[i]) 
    elif true[i]==1:
        false_lab.append(uncertainty[i]) 
    elif true[i]==2:
        true_lab.append(uncertainty[i]) 
    elif true[i]==3:
        unverified_lab.append(uncertainty[i]) 


data = [nonrum_lab, true_lab, false_lab, unverified_lab]
fig, ax = plt.subplots()
ax.boxplot(data)
#title_string = "Boxplots of NTFU "+utype+" uncertainty" 
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Non-Rumour','True','False', 'Unverified'])
#ax.set_title(title_string)

plt.show()
fig.savefig("tfun_"+utype+"_tw15_T50_w05.pdf", bbox_inches='tight')
#%%


print ("Kruskal tests for "+utype+" uncertainty, TFU")
print (kruskal(true_lab,false_lab,unverified_lab,nonrum_lab))

print ("Kruskal tests for "+utype+" uncertainty, TF")
print (kruskal(true_lab,false_lab))

print ("Kruskal tests for "+utype+" uncertainty, TU")
print (kruskal(true_lab,unverified_lab))

print ("Kruskal tests for "+utype+" uncertainty, FU")
print (kruskal(false_lab,unverified_lab))

print ("Kruskal tests for "+utype+" uncertainty, NF")
print (kruskal(nonrum_lab,false_lab))

print ("Kruskal tests for "+utype+" uncertainty, TN")
print (kruskal(true_lab,nonrum_lab))

print ("Kruskal tests for "+utype+" uncertainty, NU")
print (kruskal(nonrum_lab,unverified_lab))

#%%

correct_T = []
correct_F = []
correct_U = []
incorrect_T = []
incorrect_F = []
incorrect_U = []

for i in range(len(true)):
    if true[i]==predicted[i]:
        
        if true[i]==0:
            correct_T.append(uncertainty[i]) 
        elif true[i]==1:
            correct_F.append(uncertainty[i]) 
        elif true[i]==2:
            correct_U.append(uncertainty[i]) 

    else:
        if true[i]==0:
            incorrect_T.append(uncertainty[i]) 
        elif true[i]==1:
            incorrect_F.append(uncertainty[i]) 
        elif true[i]==2:
            incorrect_U.append(uncertainty[i]) 


data = [correct_T,correct_F,correct_U,incorrect_T,incorrect_F,incorrect_U]
#fig, ax = plt.subplots()
fig = plt.figure()  
ax = plt.axes()
ax.boxplot(data)
title_string = "Boxplots of TRUE correct,incorrect "+utype+" uncertainty" 
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_xticklabels(['Correct T','Correct F','Correct U', 'Incorrect T','Incorrect F','Incorrect U'])
ax.set_title(title_string)
fig.savefig(title_string+'.png')
#%%

correct_T = []
correct_F = []
correct_U = []
incorrect_T = []
incorrect_F = []
incorrect_U = []

for i in range(len(true)):
    if true[i]==predicted[i]:
        
        if predicted[i]==0:
            correct_T.append(uncertainty[i]) 
        elif predicted[i]==1:
            correct_F.append(uncertainty[i]) 
        elif predicted[i]==2:
            correct_U.append(uncertainty[i]) 

    else:
        if predicted[i]==0:
            incorrect_T.append(uncertainty[i]) 
        elif predicted[i]==1:
            incorrect_F.append(uncertainty[i]) 
        elif predicted[i]==2:
            incorrect_U.append(uncertainty[i]) 


data = [correct_T,correct_F,correct_U,incorrect_T,incorrect_F,incorrect_U]
#fig, ax = plt.subplots()
fig = plt.figure()  
ax = plt.axes()
ax.boxplot(data)
title_string = "Boxplots of PREDICTED correct,incorrect "+utype+" uncertainty" 
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_xticklabels(['Correct T','Correct F','Correct U', 'Incorrect T','Incorrect F','Incorrect U'])
ax.set_title(title_string)
fig.savefig(title_string+'.png')


#%%

# analysing the length of the conversation
from matplotlib import rcParams

labelsize = 18
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 
# distribution of number of branches    
fig = plt.figure()  
ax = plt.axes()
title_string = "Histogram of number of tweets per tree"+dataset  
#ax.set_title(title_string)
ax.hist(tree_n_tweets, bins=40)
fig.savefig('length_analysis/'+title_string+'.png')

#sum(np.asarray(tree_n_branches)>100) - 42
#sum(100>np.asarray(tree_n_branches)>50) - 84
#sum(np.asarray(tree_n_branches)<10) - 226
#sum(np.asarray(tree_n_branches)<5) - 107

utype='variation_ratio'

if utype=='aleatoric':
    uncertainty=aleatoric_uncertainty
elif utype=='variation_ratio':
   uncertainty=variation_ratio
elif utype=='softmax':
    uncertainty=softmax

uncertainty = np.asarray(uncertainty)
# equal depth ish for Twitter 16
#l100 = uncertainty[np.asarray(tree_n_branches)>=162]
#l50 = uncertainty[(np.asarray(tree_n_branches)>=56)&(np.asarray(tree_n_branches)<162)]
#l40 = uncertainty[(np.asarray(tree_n_branches)>=30)&(np.asarray(tree_n_branches)<56)]
#l30 = uncertainty[(np.asarray(tree_n_branches)>=20)&(np.asarray(tree_n_branches)<30)]
##l20 = uncertainty[(np.asarray(tree_n_branches)>=20)&(np.asarray(tree_n_branches)<)]
#l15 = uncertainty[(np.asarray(tree_n_branches)>=14)&(np.asarray(tree_n_branches)<20)]
#l10 = uncertainty[(np.asarray(tree_n_branches)>=9)&(np.asarray(tree_n_branches)<14)]
#l5 = uncertainty[(np.asarray(tree_n_branches)>=5)&(np.asarray(tree_n_branches)<9)]
##l4 = uncertainty[np.asarray(tree_n_branches)==4]
##l3 = uncertainty[np.asarray(tree_n_branches)==3]
##l2 = uncertainty[np.asarray(tree_n_branches)==2]
##l1 = uncertainty[np.asarray(tree_n_branches)==1]
#l0 = uncertainty[np.asarray(tree_n_branches)<5]
# equal depth ish for Twitter 16 - 8 bins
#l100 = uncertainty[np.asarray(tree_n_branches)>=60]
#l50 = uncertainty[(np.asarray(tree_n_branches)>=33)&(np.asarray(tree_n_branches)<60)]
#l40 = uncertainty[(np.asarray(tree_n_branches)>=22)&(np.asarray(tree_n_branches)<33)]
#l30 = uncertainty[(np.asarray(tree_n_branches)>=16)&(np.asarray(tree_n_branches)<22)]
##l20 = uncertainty[(np.asarray(tree_n_branches)>=20)&(np.asarray(tree_n_branches)<)]
#l15 = uncertainty[(np.asarray(tree_n_branches)>=11)&(np.asarray(tree_n_branches)<16)]
#l10 = uncertainty[(np.asarray(tree_n_branches)>=7)&(np.asarray(tree_n_branches)<11)]
#l5 = uncertainty[(np.asarray(tree_n_branches)>=4)&(np.asarray(tree_n_branches)<7)]
##l4 = uncertainty[np.asarray(tree_n_branches)==4]
##l3 = uncertainty[np.asarray(tree_n_branches)==3]
##l2 = uncertainty[np.asarray(tree_n_branches)==2]
##l1 = uncertainty[np.asarray(tree_n_branches)==1]
#l0 = uncertainty[np.asarray(tree_n_branches)<4]
# equal depth ish for Twitter 16 - 5 bins
l100 = uncertainty[np.asarray(tree_n_tweets)>=42]
l15 = uncertainty[(np.asarray(tree_n_tweets)>=21)&(np.asarray(tree_n_tweets)<42)]
l10 = uncertainty[(np.asarray(tree_n_tweets)>=13)&(np.asarray(tree_n_tweets)<21)]
l5 = uncertainty[(np.asarray(tree_n_tweets)>=7)&(np.asarray(tree_n_tweets)<13)]
l0 = uncertainty[np.asarray(tree_n_tweets)<7]
# equal depth ish for Twitter 15 - 5 bins
#l100 = uncertainty[np.asarray(tree_n_tweets)>=49]
#l15 = uncertainty[(np.asarray(tree_n_tweets)>=24)&(np.asarray(tree_n_tweets)<49)]
#l10 = uncertainty[(np.asarray(tree_n_tweets)>=14)&(np.asarray(tree_n_tweets)<24)]
#l5 = uncertainty[(np.asarray(tree_n_tweets)>=8)&(np.asarray(tree_n_tweets)<14)]
#l0 = uncertainty[np.asarray(tree_n_tweets)<8]

#data = [l1,l2,l3,l4,l5,l10,l20,l30,l40,l50,l100]
#data = [l0,l5,l10,l15,l30,l40,l50,l100]
data = [l0,l5,l10,l15,l100]
bin_size = [len(i) for i in data]

fig, ax = plt.subplots()
bp = ax.boxplot(data, notch=0, sym='', vert=1, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')
#title_string = "Boxplots uncertainty" 
# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

ax.set_xlim(0.5, len(data) + 0.5)
top = 0.65
#top = 1
#top = 0.00002
#bottom = 0.89

#top = 0.00002
bottom=0
ax.set_ylim(bottom, top)
#ax.set_xticklabels(np.repeat(randomDists, 2),rotation=45, fontsize=8)
ax.set_xlabel('Number of tweets in the tree', fontsize=18)
ax.set_ylabel('Uncertainty', fontsize=18)

# Hide these grid behind plot objects
ax.set_axisbelow(True)
ax.set_xticks(np.asarray(range(len(data)))+1)
ax.set_xticklabels(['0-6','7-12','13-20','21-41','42+'])
#ax.set_xticklabels(['0-7','8-13','14-23','24-49','49+'])
ax.set_title('Number of trees in each box', fontsize=18)
#ax.boxplot(data)
medians = bin_size
pos = np.arange(len(data)) + 1
upperLabels = [str(np.round(s, 2)) for s in medians]
weights = ['bold', 'semibold']
for tick, label in zip(range(len(data)), ax.get_xticklabels()):
    k = tick % 2
    ax.text(pos[tick], top - (top*0.107), upperLabels[tick],
             horizontalalignment='center', size='x-large', weight='semibold',
             color='black')

plt.show()    





































