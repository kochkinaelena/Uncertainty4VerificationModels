import os 
import numpy as np
import json
import gensim
import nltk
import re
from nltk.corpus import stopwords
from copy import deepcopy
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
import help_prep_functions
from lexer import lexicon_reader
#%%



def extract_thread_features(conversation):
  
    features = []
    feature_dict = {}
    
    tw = conversation['source']
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', tw['text'].lower()))
    
    otherthreadtweets = ''
    for response in conversation['replies']:
      if response['user']['screen_name'] != tw['user']['screen_name']:
        otherthreadtweets += ' ' + response['text']
    otherthreadtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', otherthreadtweets.lower()))

    raw_txt = tw['text']
    tw['text'] = help_prep_functions.cleantweet(tw['text'], tw)
    
    feature_dict['hasqmark'] = 0
    if tw['text'].find('?') >= 0:
        feature_dict['hasqmark'] = 1  
       
    feature_dict['hasemark'] = 0
    if tw['text'].find('!') >= 0:
        feature_dict['hasemark'] = 1
        
    feature_dict['hasperiod'] = 0
    if tw['text'].find('.') >= 0:
        feature_dict['hasperiod'] = 1
                    
    feature_dict['hashashtag'] = 0
    if tw['text'].find('#') >= 0:
        feature_dict['hashashtag'] = 1

    feature_dict['hasurl'] = 0
    if tw['text'].find('urlurlurl') >= 0 or tw['text'].find('http') >= 0:
        feature_dict['hasurl'] = 1

    feature_dict['haspic'] = 0
    if tw['text'].find('picpicpic') >= 0 or tw['text'].find('pic.twitter.com') >= 0 or tw['text'].find('instagr.am') >= 0:
        feature_dict['haspic'] = 1
        
    feature_dict['hasnegation'] = 0
    negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never', 'neither', 'nor', 'nowhere', 'hardly', 'scarcely', 'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn', 'couldn', 'doesn']
    for negationword in negationwords:
        if negationword in tokens:
            feature_dict['hasnegation'] += 1
    
    feature_dict['charcount'] = len(tw['text'])        
    feature_dict['wordcount'] = len(nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', tw['text'].lower())))

    swearwords = []
    with open('badwords.txt', 'r') as f:
        for line in f:
            swearwords.append(line.strip().lower())

    feature_dict['hasswearwords'] = 0
    for token in tokens:
        if token in swearwords:
            feature_dict['hasswearwords'] += 1
    
    #print raw_txt
    uppers = [l for l in raw_txt if l.isupper()]
    #print uppers
    feature_dict['capitalratio'] = float(len(uppers))/len(raw_txt)

    
    feature_dict['Word2VecSimilarityWrtOther'] = help_prep_functions.getW2vCosineSimilarity(tokens, otherthreadtokens) 
    feature_dict['Word2VecSimilarityWrtOther_PHEME'] = help_prep_functions.getW2vCosineSimilarity(tokens, otherthreadtokens, model_name='model_PHEME' ) 

    feature_dict['avgw2v'] = help_prep_functions.sumw2v(tw, avg = True,  model_name='model_GN')  
    
    feature_dict['avgw2v_PHEME'] = help_prep_functions.sumw2v(tw, avg = True,  model_name='model_PHEME')  
#    ADD Features here
#%%
    # these features are covered in lexicon features
#    negative_words = []
#    with open('negative-words.txt', 'r') as f:
#        for line in f:
#            negative_words.append(line.strip().lower().decode('utf-8'))
#  
#    positive_words = []
#    with open('positive-words.txt', 'r') as f:
#        for line in f:
#            positive_words.append(line.strip().lower().decode('utf-8'))
#            
#    feature_dict['src_numnegwords'] = 0
#    for token in tokens:
#        if token in negative_words:
#            feature_dict['src_numnegwords'] += 1
#            
#    feature_dict['src_numposwords'] = 0
#    for token in tokens:
#        if token in positive_words:
#            feature_dict['src_numposwords'] += 1
#                        
#    feature_dict['thread_numnegwords'] = 0
#    for token in alltokens:
#        if token in negative_words:
#            feature_dict['thread_numnegwords'] += 1
#                        
#    feature_dict['thread_numposwords'] = 0
#    for token in alltokens:
#        if token in positive_words:
#            feature_dict['thread_numposwords'] += 1  
                        
    #%% user based
    
    feature_dict['src_num_followers'] = tw['user']['followers_count']
    feature_dict['src_num_friends'] = tw['user']['friends_count']
    feature_dict['src_verified_user'] = int(tw['user']['verified'])
    feature_dict['src_usr_hasurl'] = 0
    if tw['user']['url'] != None:
        feature_dict['src_usr_hasurl'] = 1
                    
    feature_dict['src_utc_offset'] = -1
    if tw['user']['utc_offset'] != None:                
        feature_dict['src_utc_offset'] = tw['user']['utc_offset']
    
        
    feature_dict['src_statuses_count'] = tw['user']['statuses_count']
#    feature_dict['src_protected'] = int(tw['user']['protected'])
    # how many lists a user belongs to
    feature_dict['src_listed_count'] = tw['user']['listed_count']
#    feature_dict['src_has_description'] = 0
    feature_dict['src_description'] = np.zeros(300)            
    if tw['user']['description'] != None:
#        feature_dict['src_has_description'] = 1    
        feature_dict['src_description'] = help_prep_functions.text_sumw2v(tw['user']['description'], avg = True) 
        feature_dict['src_description_PHEME'] = help_prep_functions.text_sumw2v(tw['user']['description'], avg = True, model_name='model_PHEME')
#    else:
#        print "hey"
    
#    feature_dict['src_has_background_image'] = int(tw['user']['profile_use_background_image'])
#    feature_dict['src_has_profile_image'] = 1-int(tw['user']['default_profile_image'])
#    feature_dict['src_has_contributors'] = int(tw['user']['contributors_enabled'])
    feature_dict['src_usr_favourites_count'] = int(tw['user']['favourites_count'])
    feature_dict['src_geo_enabled'] = int(tw['user']['geo_enabled'])
    
    if feature_dict['src_num_friends']!= 0:
        feature_dict['src_follow_ratio'] = float(feature_dict['src_num_followers'])/float(feature_dict['src_num_friends'])
    else:
        feature_dict['src_follow_ratio'] = float(feature_dict['src_num_followers'])/1.0 # only 2 instances like that
#        print "1"

    acc_create_date = datetime.strptime(tw['user']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
    tw_date = datetime.strptime(tw['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
    feature_dict['account_age'] = (tw_date - acc_create_date).days
    #%%
#    feature_dict['src_has_contributors'] = 0
#    if tw['contributors'] != None:
#        feature_dict['src_has_contributors'] = 1   
                    
    feature_dict['src_has_coordinates'] =0
    if tw['coordinates']!= None:
        feature_dict['src_has_coordinates'] = 1             
             
    feature_dict['src_favourite_count'] = tw['favorite_count']
    
    feature_dict['src_retweet_count'] = tw['retweet_count']
 
    postag_tuples = nltk.pos_tag(tokens)
    postag_list = [x[1] for x in postag_tuples]
    
    possible_postags = ['WRB', 'WP$','WP', 'WDT', 'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB', 'UH', 'TO', 'SYM', 'RP', 'RBS', 'RBR', 'RB', 'PRP$', 'PRP',  'POS', 'PDT', 'NNS', 'NNPS', 'NNP', 'NN', 'MD', 'LS', 'JJS', 'JJR', 'JJ', 'IN', 'FW', 'EX', 'DT', 'CD', 'CC', '$'] 
        
    postag_binary = np.zeros(len(possible_postags))
    
    for tok in postag_list:
        postag_binary[possible_postags.index(tok)] = 1
    
    feature_dict['pos'] = postag_binary
    
    
    lexicon = lexicon_reader() 
    feature_dict['src_lex'] = lexicon.get_lexicon_scores(tokens)
#    feature_dict['thread_lex'] =  lexicon.get_lexicon_scores(otherthreadtokens)
    
    false_synonyms = ['false',  'bogus',  'deceitful',  'dishonest',  'distorted',  'erroneous',  'fake','fanciful',  'faulty',  'fictitious',  'fraudulent',  
                        'improper',  'inaccurate',  'incorrect',  'invalid', 'misleading', 'mistaken', 'phony', 'specious', 'spurious', 'unfounded', 'unreal',
                        'untrue',  'untruthful',  'apocryphal',  'beguiling',  'casuistic',  'concocted', 'cooked-up', 'counterfactual', 
                        'deceiving', 'delusive', 'ersatz', 'fallacious','fishy',  'illusive',  'imaginary',  'inexact',  'lying',  'mendacious',  
                        'misrepresentative', 'off the mark', 'sham', 'sophistical', 'trumped up', 'unsound']
    
    false_antonyms = ['accurate', 'authentic', 'correct', 'fair', 'faithful', 'frank', 'genuine', 'honest', 'moral', 'open', 'proven', 'real', 'right', 'sincere', 'sound', 'true', 
                      'trustworthy', 'truthful', 'valid', 'actual', 'factual', 'just', 'known', 'precise', 'reliable', 'straight', 'substantiated']
    
    
    feature_dict['src_num_false_synonyms'] = 0
    for token in tokens:
        if token in false_synonyms:
            feature_dict['src_num_false_synonyms'] += 1
                        
    feature_dict['src_num_false_antonyms'] = 0
    for token in tokens:
        if token in false_antonyms:
            feature_dict['src_num_false_antonyms'] += 1
                        
    feature_dict['thread_num_false_synonyms'] = 0
    for token in otherthreadtokens:
        if token in false_synonyms:
            feature_dict['thread_num_false_synonyms'] += 1
                        
    feature_dict['thread_num_false_antonyms'] = 0
    for token in otherthreadtokens:
        if token in false_antonyms:
            feature_dict['thread_num_false_antonyms'] += 1
    
    feature_dict['src_unconfirmed'] = 0
    feature_dict['src_rumour'] = 0
                
    feature_dict['thread_unconfirmed'] = 0
    feature_dict['thread_rumour'] = 0
                
    if 'unconfirmed' in tokens:
        feature_dict['src_unconfirmed'] = 1
                    
    if 'unconfirmed' in otherthreadtokens:
        feature_dict['thread_unconfirmed'] = 1
                    
                    
                    
    if 'rumour' in tokens  or 'gossip' in tokens or  'hoax' in tokens :
        feature_dict['src_rumour'] = 1    
                    
    if 'rumour' in otherthreadtokens or 'gossip' in otherthreadtokens or 'hoax' in otherthreadtokens:
        feature_dict['thread_rumour'] = 1    
                    
                    
    whwords = ['what', 'when','where','which','who','whom','whose','why','how']
    
    feature_dict['src_num_wh'] = 0
    for token in tokens:
        if token in whwords:
            feature_dict['src_num_wh'] += 1
                        
    feature_dict['thread_num_wh'] = 0
    for token in otherthreadtokens:
        if token in whwords:
            feature_dict['thread_num_wh'] += 1
                    
    SpeechAct = {}                    
    SpeechAct['SpeechAct_ORDER'] = ['command', 'demand', 'tell', 'direct', 'instruct', 'require', 'prescribe', 'order']     
    SpeechAct['SpeechAct_ASK1'] = ['ask','request','beg','bespeech','implore','appeal', 'plead', 'intercede', 'apply', 'urge', 'persuade', 'dissuade', 'convince']
    SpeechAct['SpeechAct_ASK2'] = ['ask', 'inquire', 'enquire', 'interrogate', 'question', 'query']
    SpeechAct['SpeechAct_CALL'] = ['call', 'summon', 'invite', 'call on', 'call for', 'order', 'book', 'reserve']
    SpeechAct['SpeechAct_FORBID'] = ['forbid', 'prohibit', 'veto', 'refuse', 'decline', 'reject', 'rebuff', 'renounce', 'cancel', 'resign', 'dismiss']
    SpeechAct['SpeechAct_PERMIT'] = ['permit', 'allow', 'consent', 'accept', 'agree', 'approve', 'disapprove', 'authorize', 'appoint']
    SpeechAct['SpeechAct_ARGUE'] = ['argue', 'disagree', 'refute', 'contradict', 'counter', 'deny', 'recant', 'retort', 'quarrel']
    SpeechAct['SpeechAct_REPRIMAND' ]= ['reprimand', 'rebuke', 'reprove', 'admonish', 'reproach', 'nag', 'scold', 'abuse', 'insult']
    SpeechAct['SpeechAct_MOCK'] = ['ridicule', 'joke']
    SpeechAct['SpeechAct_BLAME'] = ['blame', 'criticize', 'condemn', 'denounce', 'deplore', 'curse']
    SpeechAct['SpeechAct_ACCUSE'] = ['accuse', 'charge', 'challenge', 'defy', 'dare']
    SpeechAct['SpeechAct_ATTACK'] = ['attack', 'defend']
    SpeechAct['SpeechAct_WARN ']= ['warn', 'threaten', 'blackmail']
    SpeechAct['SpeechAct_ADVISE ']= ['advise', 'councel', 'consult', 'recommend', 'suggest', 'propose', 'advocate']
    SpeechAct['SpeechAct_OFFER ']= ['offer', 'volunteer', 'grant', 'give']
    SpeechAct['SpeechAct_PRAISE ']= ['praise', 'commend', 'compliment', 'boast', 'credit']                  
    SpeechAct['SpeechAct_PROMISE ']= ['promise', 'pledge', 'vow', 'swear', 'vouch for', 'guarante']    
    SpeechAct['SpeechAct_THANK ']= ['thank', 'apologise', 'greet', 'welcome', 'farewell', 'goodbye', 'introduce', 'bless','wish', 'congratulate']
    SpeechAct['SpeechAct_FORGIVE ']= ['forgive', 'excuse', 'justify', 'absolve', 'pardon', 'convict', 'acquit', 'sentence']
    SpeechAct['SpeechAct_COMPLAIN'] = ['complain', 'protest', 'object', 'moan', 'bemoan','lament', 'bewail']
    
    SpeechAct['SpeechAct_EXCLAIM'] = ['exclaim', 'enthuse', 'exult', 'swear', 'blaspheme']
    SpeechAct['SpeechAct_GUESS']= ['guess', 'bet', 'presume','suspect', 'suppose', 'wonder', 'speculate', 'conjecture', 'predict', 'forecast', 'prophesy']
    SpeechAct['SpeechAct_HINT']= ['hint','imply','insinuate']
    SpeechAct['SpeechAct_CONCLUDE']= ['conclude', 'deduce', 'infer','gather', 'reckon', 'estimate', 'calculate', 'count','prove', 'compare']
    SpeechAct['SpeechAct_TELL']= ['tell', 'report', 'narrate', 'relate','recount', 'describe', 'explain', 'lecture']
    SpeechAct['SpeechAct_INFORM']= ['inform', 'notify', 'announce', 'inform on', 'reveal']
    SpeechAct['SpeechAct_SUMUP']= ['sum up', 'summarize', 'recapitulate']
    SpeechAct['SpeechAct_ADMIT']= ['admit', 'acknowledge', 'concede','confess', 'confide']
    SpeechAct['SpeechAct_ASSERT']= ['assert', 'affirm', 'claim', 'maintain', 'contend', 'state','testify']
    SpeechAct['SpeechAct_CONFIRM']= ['confirm', 'assure','reassure']
    SpeechAct['SpeechAct_STRESS']= [' stress','emphasize', 'insist', 'repeat', 'point out', 'note', 'remind', 'add' ]
    SpeechAct['SpeechAct_DECLARE']= ['declare', 'pronounce', 'proclaim', 'decree', 'profess', 'vote', 'resolve', 'decide']
    SpeechAct['SpeechAct_BAPTIZE']= ['baptize', 'chirsten', 'name', 'excommunicate']
    SpeechAct['SpeechAct_REMARK']= ['remark', 'comment', 'observe']
    SpeechAct['SpeechAct_ANSWER']  = ['answer', 'reply']
    SpeechAct['SpeechAct_DISCUSS']= ['discuss', 'debate', 'negotiate', 'bargain']
    SpeechAct['SpeechAct_TALK']= ['talk', 'converse', 'chat', 'gossip']
    
    for k in SpeechAct.keys():
        feature_dict[k] = 0
        for verb in SpeechAct[k]:
            if verb in tw['text'].lower():
                feature_dict[k] += 1
                            

        
    return feature_dict


def extract_thread_features_incl_response(conversation):
#%%
    
    source_features = extract_thread_features(conversation)
    
    fullthread_featdict = {}
    fullthread_featdict[conversation['source']['id_str']] = source_features
    
    for tw in conversation['replies']: 
        features = []
        feature_dict = {}
    
        tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', tw['text'].lower()))
    
        otherthreadtweets = ''
        if conversation['source']['user']['screen_name'] != tw['user']['screen_name']:
            otherthreadtweets += conversation['source']['text']
        for response in conversation['replies']:
          if response['user']['screen_name'] != tw['user']['screen_name']:
            otherthreadtweets += ' ' + response['text']
        
        otherthreadtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', otherthreadtweets.lower()))
    
        alltokens = tokens + otherthreadtokens
        raw_txt = tw['text']
        tw['text'] = help_prep_functions.cleantweet(tw['text'], tw)
        
        feature_dict['hasqmark'] = 0
        if tw['text'].find('?') >= 0:
            feature_dict['hasqmark'] = 1  
           
        feature_dict['hasemark'] = 0
        if tw['text'].find('!') >= 0:
            feature_dict['hasemark'] = 1
            
        feature_dict['hasperiod'] = 0
        if tw['text'].find('.') >= 0:
            feature_dict['hasperiod'] = 1
                        
        feature_dict['hashashtag'] = 0
        if tw['text'].find('#') >= 0:
            feature_dict['hashashtag'] = 1
    
        feature_dict['hasurl'] = 0
        if tw['text'].find('urlurlurl') >= 0 or tw['text'].find('http') >= 0:
            feature_dict['hasurl'] = 1
    
        feature_dict['haspic'] = 0
        if tw['text'].find('picpicpic') >= 0 or tw['text'].find('pic.twitter.com') >= 0 or tw['text'].find('instagr.am') >= 0:
            feature_dict['haspic'] = 1
            
        feature_dict['hasnegation'] = 0
        negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never', 'neither', 'nor', 'nowhere', 'hardly', 'scarcely', 'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn', 'couldn', 'doesn']
        for negationword in negationwords:
            if negationword in tokens:
                feature_dict['hasnegation'] += 1
        
        feature_dict['charcount'] = len(tw['text'])        
        feature_dict['wordcount'] = len(nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '', tw['text'].lower())))
    
        swearwords = []
        with open('badwords.txt', 'r') as f:
            for line in f:
                swearwords.append(line.strip().lower())
    
        feature_dict['hasswearwords'] = 0
        for token in tokens:
            if token in swearwords:
                feature_dict['hasswearwords'] += 1
        
        #print raw_txt
        uppers = [l for l in raw_txt if l.isupper()]
        #print uppers
        feature_dict['capitalratio'] = float(len(uppers))/len(raw_txt)
    
        
        feature_dict['Word2VecSimilarityWrtOther'] = help_prep_functions.getW2vCosineSimilarity(tokens, otherthreadtokens) 
        feature_dict['Word2VecSimilarityWrtOther_PHEME'] = help_prep_functions.getW2vCosineSimilarity(tokens, otherthreadtokens, model_name='model_PHEME' ) 
    
        feature_dict['avgw2v'] = help_prep_functions.sumw2v(tw, avg = True,  model_name='model_GN')  
        
        feature_dict['avgw2v_PHEME'] = help_prep_functions.sumw2v(tw, avg = True,  model_name='model_PHEME')  
#    ADD Features here
#%%
    # these features are covered in lexicon features
#    negative_words = []
#    with open('negative-words.txt', 'r') as f:
#        for line in f:
#            negative_words.append(line.strip().lower().decode('utf-8'))
#  
#    positive_words = []
#    with open('positive-words.txt', 'r') as f:
#        for line in f:
#            positive_words.append(line.strip().lower().decode('utf-8'))
#            
#    feature_dict['src_numnegwords'] = 0
#    for token in tokens:
#        if token in negative_words:
#            feature_dict['src_numnegwords'] += 1
#            
#    feature_dict['src_numposwords'] = 0
#    for token in tokens:
#        if token in positive_words:
#            feature_dict['src_numposwords'] += 1
#                        
#    feature_dict['thread_numnegwords'] = 0
#    for token in alltokens:
#        if token in negative_words:
#            feature_dict['thread_numnegwords'] += 1
#                        
#    feature_dict['thread_numposwords'] = 0
#    for token in alltokens:
#        if token in positive_words:
#            feature_dict['thread_numposwords'] += 1  
                        
    #%% user based
    
        feature_dict['src_num_followers'] = tw['user']['followers_count']
        feature_dict['src_num_friends'] = tw['user']['friends_count']
        feature_dict['src_verified_user'] = int(tw['user']['verified'])
        feature_dict['src_usr_hasurl'] = 0
        if tw['user']['url'] != None:
            feature_dict['src_usr_hasurl'] = 1
                        
        feature_dict['src_utc_offset'] = -1
        if tw['user']['utc_offset'] != None:                
            feature_dict['src_utc_offset'] = tw['user']['utc_offset']
        
            
        feature_dict['src_statuses_count'] = tw['user']['statuses_count']
    #    feature_dict['src_protected'] = int(tw['user']['protected'])
        # how many lists a user belongs to
        feature_dict['src_listed_count'] = tw['user']['listed_count']
    #    feature_dict['src_has_description'] = 0
        feature_dict['src_description'] = np.zeros(300)            
        if tw['user']['description'] != None:
    #        feature_dict['src_has_description'] = 1    
            feature_dict['src_description'] = help_prep_functions.text_sumw2v(tw['user']['description'], avg = True) 
            feature_dict['src_description_PHEME'] = help_prep_functions.text_sumw2v(tw['user']['description'], avg = True, model_name='model_PHEME')
    #    else:
    #        print "hey"
        
    #    feature_dict['src_has_background_image'] = int(tw['user']['profile_use_background_image'])
    #    feature_dict['src_has_profile_image'] = 1-int(tw['user']['default_profile_image'])
    #    feature_dict['src_has_contributors'] = int(tw['user']['contributors_enabled'])
        feature_dict['src_usr_favourites_count'] = int(tw['user']['favourites_count'])
        feature_dict['src_geo_enabled'] = int(tw['user']['geo_enabled'])
        
        if feature_dict['src_num_friends']!= 0:
            feature_dict['src_follow_ratio'] = float(feature_dict['src_num_followers'])/float(feature_dict['src_num_friends'])
        else:
            feature_dict['src_follow_ratio'] = float(feature_dict['src_num_followers'])/1.0 # only 2 instances like that
    #        print "1"
    
        acc_create_date = datetime.strptime(tw['user']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        tw_date = datetime.strptime(tw['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        feature_dict['account_age'] = (tw_date - acc_create_date).days
        #%%
    #    feature_dict['src_has_contributors'] = 0
    #    if tw['contributors'] != None:
    #        feature_dict['src_has_contributors'] = 1   
                        
        feature_dict['src_has_coordinates'] =0
        if tw['coordinates']!= None:
            feature_dict['src_has_coordinates'] = 1             
                 
        feature_dict['src_favourite_count'] = tw['favorite_count']
        
        feature_dict['src_retweet_count'] = tw['retweet_count']
     
        postag_tuples = nltk.pos_tag(tokens)
        postag_list = [x[1] for x in postag_tuples]
        
        possible_postags = ['WRB', 'WP$','WP', 'WDT', 'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB', 'UH', 'TO', 'SYM', 'RP', 'RBS', 'RBR', 'RB', 'PRP$', 'PRP',  'POS', 'PDT', 'NNS', 'NNPS', 'NNP', 'NN', 'MD', 'LS', 'JJS', 'JJR', 'JJ', 'IN', 'FW', 'EX', 'DT', 'CD', 'CC', '$'] 
        
        postag_binary = np.zeros(len(possible_postags))
        
#        print postag_list
        for tok in postag_list:
            if tok in possible_postags:
                postag_binary[possible_postags.index(tok)] = 1
        
        feature_dict['pos'] = postag_binary
        
        
        lexicon = lexicon_reader() 
        feature_dict['src_lex'] = lexicon.get_lexicon_scores(tokens)
#        feature_dict['thread_lex'] =  lexicon.get_lexicon_scores(otherthreadtokens)
        
        false_synonyms = ['false',  'bogus',  'deceitful',  'dishonest',  'distorted',  'erroneous',  'fake','fanciful',  'faulty',  'fictitious',  'fraudulent',  
                            'improper',  'inaccurate',  'incorrect',  'invalid', 'misleading', 'mistaken', 'phony', 'specious', 'spurious', 'unfounded', 'unreal',
                            'untrue',  'untruthful',  'apocryphal',  'beguiling',  'casuistic',  'concocted', 'cooked-up', 'counterfactual', 
                            'deceiving', 'delusive', 'ersatz', 'fallacious','fishy',  'illusive',  'imaginary',  'inexact',  'lying',  'mendacious',  
                            'misrepresentative', 'off the mark', 'sham', 'sophistical', 'trumped up', 'unsound']
        
        false_antonyms = ['accurate', 'authentic', 'correct', 'fair', 'faithful', 'frank', 'genuine', 'honest', 'moral', 'open', 'proven', 'real', 'right', 'sincere', 'sound', 'true', 
                          'trustworthy', 'truthful', 'valid', 'actual', 'factual', 'just', 'known', 'precise', 'reliable', 'straight', 'substantiated']
        
        
        feature_dict['src_num_false_synonyms'] = 0
        for token in tokens:
            if token in false_synonyms:
                feature_dict['src_num_false_synonyms'] += 1
                            
        feature_dict['src_num_false_antonyms'] = 0
        for token in tokens:
            if token in false_antonyms:
                feature_dict['src_num_false_antonyms'] += 1
                            
        feature_dict['thread_num_false_synonyms'] = 0
        for token in otherthreadtokens:
            if token in false_synonyms:
                feature_dict['thread_num_false_synonyms'] += 1
                            
        feature_dict['thread_num_false_antonyms'] = 0
        for token in otherthreadtokens:
            if token in false_antonyms:
                feature_dict['thread_num_false_antonyms'] += 1
        
        feature_dict['src_unconfirmed'] = 0
        feature_dict['src_rumour'] = 0
                    
        feature_dict['thread_unconfirmed'] = 0
        feature_dict['thread_rumour'] = 0
                    
        if 'unconfirmed' in tokens:
            feature_dict['src_unconfirmed'] = 1
                        
        if 'unconfirmed' in otherthreadtokens:
            feature_dict['thread_unconfirmed'] = 1
                        
                        
                        
        if 'rumour' in tokens  or 'gossip' in tokens or  'hoax' in tokens :
            feature_dict['src_rumour'] = 1    
                        
        if 'rumour' in otherthreadtokens or 'gossip' in otherthreadtokens or 'hoax' in otherthreadtokens:
            feature_dict['thread_rumour'] = 1    
                        
                        
        whwords = ['what', 'when','where','which','who','whom','whose','why','how']
        
        feature_dict['src_num_wh'] = 0
        for token in tokens:
            if token in whwords:
                feature_dict['src_num_wh'] += 1
                            
        feature_dict['thread_num_wh'] = 0
        for token in otherthreadtokens:
            if token in whwords:
                feature_dict['thread_num_wh'] += 1
                        
        SpeechAct = {}                    
        SpeechAct['SpeechAct_ORDER'] = ['command', 'demand', 'tell', 'direct', 'instruct', 'require', 'prescribe', 'order']     
        SpeechAct['SpeechAct_ASK1'] = ['ask','request','beg','bespeech','implore','appeal', 'plead', 'intercede', 'apply', 'urge', 'persuade', 'dissuade', 'convince']
        SpeechAct['SpeechAct_ASK2'] = ['ask', 'inquire', 'enquire', 'interrogate', 'question', 'query']
        SpeechAct['SpeechAct_CALL'] = ['call', 'summon', 'invite', 'call on', 'call for', 'order', 'book', 'reserve']
        SpeechAct['SpeechAct_FORBID'] = ['forbid', 'prohibit', 'veto', 'refuse', 'decline', 'reject', 'rebuff', 'renounce', 'cancel', 'resign', 'dismiss']
        SpeechAct['SpeechAct_PERMIT'] = ['permit', 'allow', 'consent', 'accept', 'agree', 'approve', 'disapprove', 'authorize', 'appoint']
        SpeechAct['SpeechAct_ARGUE'] = ['argue', 'disagree', 'refute', 'contradict', 'counter', 'deny', 'recant', 'retort', 'quarrel']
        SpeechAct['SpeechAct_REPRIMAND' ]= ['reprimand', 'rebuke', 'reprove', 'admonish', 'reproach', 'nag', 'scold', 'abuse', 'insult']
        SpeechAct['SpeechAct_MOCK'] = ['ridicule', 'joke']
        SpeechAct['SpeechAct_BLAME'] = ['blame', 'criticize', 'condemn', 'denounce', 'deplore', 'curse']
        SpeechAct['SpeechAct_ACCUSE'] = ['accuse', 'charge', 'challenge', 'defy', 'dare']
        SpeechAct['SpeechAct_ATTACK'] = ['attack', 'defend']
        SpeechAct['SpeechAct_WARN ']= ['warn', 'threaten', 'blackmail']
        SpeechAct['SpeechAct_ADVISE ']= ['advise', 'councel', 'consult', 'recommend', 'suggest', 'propose', 'advocate']
        SpeechAct['SpeechAct_OFFER ']= ['offer', 'volunteer', 'grant', 'give']
        SpeechAct['SpeechAct_PRAISE ']= ['praise', 'commend', 'compliment', 'boast', 'credit']                  
        SpeechAct['SpeechAct_PROMISE ']= ['promise', 'pledge', 'vow', 'swear', 'vouch for', 'guarante']    
        SpeechAct['SpeechAct_THANK ']= ['thank', 'apologise', 'greet', 'welcome', 'farewell', 'goodbye', 'introduce', 'bless','wish', 'congratulate']
        SpeechAct['SpeechAct_FORGIVE ']= ['forgive', 'excuse', 'justify', 'absolve', 'pardon', 'convict', 'acquit', 'sentence']
        SpeechAct['SpeechAct_COMPLAIN'] = ['complain', 'protest', 'object', 'moan', 'bemoan','lament', 'bewail']
        
        SpeechAct['SpeechAct_EXCLAIM'] = ['exclaim', 'enthuse', 'exult', 'swear', 'blaspheme']
        SpeechAct['SpeechAct_GUESS']= ['guess', 'bet', 'presume','suspect', 'suppose', 'wonder', 'speculate', 'conjecture', 'predict', 'forecast', 'prophesy']
        SpeechAct['SpeechAct_HINT']= ['hint','imply','insinuate']
        SpeechAct['SpeechAct_CONCLUDE']= ['conclude', 'deduce', 'infer','gather', 'reckon', 'estimate', 'calculate', 'count','prove', 'compare']
        SpeechAct['SpeechAct_TELL']= ['tell', 'report', 'narrate', 'relate','recount', 'describe', 'explain', 'lecture']
        SpeechAct['SpeechAct_INFORM']= ['inform', 'notify', 'announce', 'inform on', 'reveal']
        SpeechAct['SpeechAct_SUMUP']= ['sum up', 'summarize', 'recapitulate']
        SpeechAct['SpeechAct_ADMIT']= ['admit', 'acknowledge', 'concede','confess', 'confide']
        SpeechAct['SpeechAct_ASSERT']= ['assert', 'affirm', 'claim', 'maintain', 'contend', 'state','testify']
        SpeechAct['SpeechAct_CONFIRM']= ['confirm', 'assure','reassure']
        SpeechAct['SpeechAct_STRESS']= [' stress','emphasize', 'insist', 'repeat', 'point out', 'note', 'remind', 'add' ]
        SpeechAct['SpeechAct_DECLARE']= ['declare', 'pronounce', 'proclaim', 'decree', 'profess', 'vote', 'resolve', 'decide']
        SpeechAct['SpeechAct_BAPTIZE']= ['baptize', 'chirsten', 'name', 'excommunicate']
        SpeechAct['SpeechAct_REMARK']= ['remark', 'comment', 'observe']
        SpeechAct['SpeechAct_ANSWER']  = ['answer', 'reply']
        SpeechAct['SpeechAct_DISCUSS']= ['discuss', 'debate', 'negotiate', 'bargain']
        SpeechAct['SpeechAct_TALK']= ['talk', 'converse', 'chat', 'gossip']
        
        for k in SpeechAct.keys():
            feature_dict[k] = 0
            for verb in SpeechAct[k]:
                if verb in tw['text'].lower():
                    feature_dict[k] += 1
                                   
    #%%
        
        fullthread_featdict[tw['id_str']] = feature_dict
    
#    list_feats = ['avgw2v', 'src_description', 'src_lex', 'thread_lex', 'pos' ] 
#                 
#    if feature_set=='Full':
#        
#        for f in feature_dict.keys():
#            if f not in list_feats:
#                features.append(feature_dict[f])
#            else:
#                features.extend(feature_dict[f])
#    else:
#        for f in feature_set:
#            if f not in list_feats:
#                features.append(feature_dict[f])
#            else:
#                features.extend(feature_dict[f])
#                
#    
#    
#    features = np.asarray(features, dtype = np.float32)  
#    if np.isnan(features).any():
#        print list(np.where(np.isnan(features)))
        
    return fullthread_featdict

