import json
import tweepy
import sys
#import pprint
#import os
import configparser as ConfigParser
#import time
#%%

def download_tweet_list(tweetid, path):
    
#%%
    config = ConfigParser.ConfigParser()
    config.read('twitter.ini')
    
    consumer_key = config.get('Twitter', 'consumer_key')
    consumer_secret = config.get('Twitter', 'consumer_secret')
    access_key = config.get('Twitter', 'access_key')
    access_secret = config.get('Twitter', 'access_secret')
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
#%%
    try:
        ltweets = api._statuses_lookup(id=tweetid, map=False)
        for tweet in ltweets:
            filename = path+str(tweet._json['id'])+'.json'
#            print filename
            with open(filename, 'w') as outfile:
                json.dump(tweet._json, outfile)
#            print json.dumps(tweet.json)
#            print tweet
         
    except:
        sys.exit()

def download_tweet(tweetid, path):
    
#%%
    config = ConfigParser.ConfigParser()
    config.read('twitter.ini')
    
    consumer_key = config.get('Twitter', 'consumer_key')
    consumer_secret = config.get('Twitter', 'consumer_secret')
    access_key = config.get('Twitter', 'access_key')
    access_secret = config.get('Twitter', 'access_secret')
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
#%%
    try:
        tweet = api.get_status(tweetid)
        filename = path+str(tweetid)+'.json'
        with open(filename, 'w') as outfile:
            json.dump(tweet, outfile)
#            print json.dumps(tweet.json)
#            print tweet
         
    except:
        sys.exit()
