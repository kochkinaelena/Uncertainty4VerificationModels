from Download_tweet_list import download_tweet_list
#import subprocess
import os
import pickle
#%%

def listdir_nohidden(path):
    files = os.listdir(path)
    newfiles = [i for i in files if i[0] != '.']
    return newfiles

def download_twitter_dataset(savepath):

    threads = listdir_nohidden(savepath)
    for t in threads:
        print (t)
        
        with open(os.path.join(savepath,t,'tweets.pkl'), 'rb') as f:
            tweets = pickle.load(f)
            
        path  = os.path.join(savepath,t, 'tweets_folder/')
        
        if not os.path.exists(path):
            os.mkdir(path)
        
        if len(tweets)<=100:
            tweetid = str(tweets)
            tweetid = tweetid[1:-1]
            tweetid = tweetid.replace(" ", "")
            tweetid = tweetid.replace("'", "")
            download_tweet_list(tweetid, path)
        else:
            n = int(len(tweets)/100)
            for i in range(n):
                tweetid = str(tweets[i*100:i*100+100])
                tweetid = tweetid[1:-1]
                tweetid = tweetid.replace(" ", "")
                tweetid = tweetid.replace("'", "")
                download_tweet_list(tweetid, path)
                
            tweetid = tweets[(i+1)*100:]
            download_tweet_list(tweetid, path)
#%%
                
savepath15 = "preprocessed_data/twitter15"
savepath16 = "preprocessed_data/twitter16"

download_twitter_dataset(savepath15)
download_twitter_dataset(savepath16)
