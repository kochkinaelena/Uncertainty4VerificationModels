
def generate_feature_set_list(text = ['avgw2v']):
    
    
    
    punctuation = ['hasqmark','hasemark','hasperiod','hashashtag','charcount','wordcount','capitalratio','pos']
    attachments = ['hasurl','src_has_coordinates','haspic']
    
    lexicon = ['hasnegation','src_lex','src_num_false_synonyms','src_num_false_antonyms','hasswearwords',
    'src_unconfirmed', 'src_rumour', 'src_num_wh' ]

    
    user = ['src_num_followers','src_num_friends','src_verified_user','src_usr_hasurl',
            'src_utc_offset','src_statuses_count','src_listed_count','src_description',
            'src_usr_favourites_count','src_geo_enabled','src_follow_ratio','account_age']
    
    interactions = ['src_favourite_count','src_retweet_count']
    stance = ['stance']
    # Change for RumourEval
#    stance = ['support_stanceratio', 'deny_stanceratio','question_stanceratio']
    
    thread = ['Word2VecSimilarityWrtOther', 'Word2VecSimilarityWrtSource', 'Word2VecSimilarityWrtPrev',
              'thread_lex', 'thread_num_false_synonyms',
              'thread_num_false_antonyms','thread_num_wh','thread_unconfirmed','thread_rumour']
    
    src_SpeechAct = ['SpeechAct_ACCUSE', 'SpeechAct_MOCK', 'SpeechAct_BAPTIZE', 'SpeechAct_CONCLUDE', 'SpeechAct_ARGUE',
                     'SpeechAct_DECLARE', 'SpeechAct_INFORM', 'SpeechAct_COMPLAIN', 'SpeechAct_BLAME',
                     'SpeechAct_REPRIMAND', 'SpeechAct_EXCLAIM', 'SpeechAct_THANK ', 'SpeechAct_GUESS',
                     'SpeechAct_PERMIT', 'SpeechAct_ATTACK', 'SpeechAct_HINT', 'SpeechAct_CALL',
                     'SpeechAct_ADMIT', 'SpeechAct_ASSERT', 'SpeechAct_ASK2', 'SpeechAct_STRESS',
                     'SpeechAct_ASK1', 'SpeechAct_PRAISE ', 'SpeechAct_WARN ', 'SpeechAct_ANSWER',
                     'SpeechAct_FORGIVE ', 'SpeechAct_OFFER ', 'SpeechAct_TALK', 'SpeechAct_PROMISE ',
                     'SpeechAct_DISCUSS', 'SpeechAct_REMARK', 'SpeechAct_TELL', 'SpeechAct_ORDER',
                     'SpeechAct_ADVISE ', 'SpeechAct_CONFIRM', 'SpeechAct_FORBID', 'SpeechAct_SUMUP']
    
    thread_SpeechAct = ['thread_SpeechAct_CALL','thread_SpeechAct_FORGIVE ','thread_SpeechAct_ANSWER','thread_SpeechAct_ORDER', 
                        'thread_SpeechAct_ADVISE ', 'thread_SpeechAct_FORBID', 'thread_SpeechAct_PRAISE ', 'thread_SpeechAct_STRESS',
                        'thread_SpeechAct_OFFER ', 'thread_SpeechAct_ATTACK', 'thread_SpeechAct_TELL', 'thread_SpeechAct_REMARK', 
                        'thread_SpeechAct_PROMISE ', 'thread_SpeechAct_REPRIMAND', 'thread_SpeechAct_PERMIT',
                        'thread_SpeechAct_DECLARE', 'thread_SpeechAct_ASK2', 'thread_SpeechAct_ASK1', 'thread_SpeechAct_EXCLAIM',
                         'thread_SpeechAct_GUESS', 'thread_SpeechAct_ARGUE', 'thread_SpeechAct_MOCK', 'thread_SpeechAct_CONCLUDE',
                         'thread_SpeechAct_SUMUP', 'thread_SpeechAct_TALK', 'thread_SpeechAct_CONFIRM', 'thread_SpeechAct_THANK ',
                         'thread_SpeechAct_WARN ', 'thread_SpeechAct_ACCUSE', 'thread_SpeechAct_INFORM', 'thread_SpeechAct_HINT',
                         'thread_SpeechAct_DISCUSS', 'thread_SpeechAct_BLAME', 'thread_SpeechAct_ADMIT','thread_SpeechAct_COMPLAIN',
                         'thread_SpeechAct_BAPTIZE', 'thread_SpeechAct_ASSERT']
    
    stance_task_features =  [  'avgw2v', 'hasnegation', 'hasswearwords', 'capitalratio', 'hasperiod', 
                               'hasqmark', 'hasemark', 'hasurl', 'haspic','charcount','wordcount', 'issource', 
                               'Word2VecSimilarityWrtOther', 'Word2VecSimilarityWrtSource', 'Word2VecSimilarityWrtPrev']
#    
#    feature_set_list = [ text+stance, text+attachments+stance, text+attachments+stance+interactions,
#                        text+punctuation+attachments+interactions+stance+thread]
#    
#    feature_name_list = [ 'text+stance', 'text+attachments+stance', 'text+attachments+stance+interactions',
#                        'text+punctuation+attachments+interactions+stance+thread']
    feature_set_list = [text]
    feature_name_list = ['avgw2v_text']
    return feature_set_list, feature_name_list


