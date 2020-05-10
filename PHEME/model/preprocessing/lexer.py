#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#also works in python 2.7 if remode 'encoding' =  from file open
import numpy as np


class lexicon_reader:
    def __init__(self):
        self.folder = 'Lexicons/lexicons/'
        
        #word->{+1/-1}
        self.afinn = self.read_AFINN()
        self.gi = self.read_general_inquirer()
        self.mpqa = self.read_mpqa()
        self.opinion_finder = self.read_opinion_finder()
        
        #word->{1/0}
        self.anger, self.anticipation, self.disgust, self.fear, self.joy, self.negative, self.positive, self.sadness, self.surprise, self.trust = self.read_nrc_wordlevel_emotions()

        #word->{score}
        self.unknown= self.read_lexicon_unknown()
        self.uni_hashtag = self.read_unigram_pmi_hashtag()
        self.uni_sentiment = self.read_unigram_pmi_sentiment140()
        self.anticipation2, self.fear2, self.anger2, self.trust2, self.surprise2, self.sadness2, self.joy2, self.disgust2 = self.read_nrc_emotions()
     
    def get_lexicon_scores(self, doc):
        
        af = self.extract_binary(doc, self.afinn)
        gi = self.extract_binary(doc, self.gi)
        mp = self.extract_binary(doc, self.mpqa)
        of = self.extract_binary(doc, self.opinion_finder)
        
        ang = self.extract_binary_emos(doc, self.anger)
        ant = self.extract_binary_emos(doc, self.anticipation)
        dis = self.extract_binary_emos(doc, self.disgust)
        fea = self.extract_binary_emos(doc, self.fear) 
        joy = self.extract_binary_emos(doc, self.joy) 
        neg = self.extract_binary_emos(doc, self.negative) 
        pos = self.extract_binary_emos(doc, self.positive) 
        sad = self.extract_binary_emos(doc, self.sadness) 
        sur = self.extract_binary_emos(doc, self.surprise) 
        tru = self.extract_binary_emos(doc, self.trust)
        
        un = self.extract_float(doc, self.unknown)
        has = self.extract_float(doc, self.uni_hashtag)
        sen = self.extract_float(doc, self.uni_sentiment)
        ang2 = self.extract_float(doc, self.anger2) 
        ant2 = self.extract_float(doc, self.anticipation2) 
        dis2 = self.extract_float(doc, self.disgust2) 
        fea2 = self.extract_float(doc, self.fear2) 
        joy2 = self.extract_float(doc, self.joy2) 
        sad2 = self.extract_float(doc, self.sadness2) 
        sur2 = self.extract_float(doc, self.surprise2) 
        tru2 = self.extract_float(doc, self.trust2)
        
        binaries = [af[0],af[1],gi[0],gi[1],mp[0],mp[1],of[0],of[1],ang,ant,dis,fea,joy,neg,pos,sad,sur,tru]
        floats = [un,has,sen,ang2,ant2,dis2,fea2,joy2,sad2,sur2,tru2]
        return [af[0],af[1],gi[0],gi[1],mp[0],mp[1],of[0],of[1],ang,ant,dis,fea,joy,neg,pos,sad,sur,tru,un,has,sen,ang2,ant2,dis2,fea2,joy2,sad2,sur2,tru2]
    
    def extract_binary(self, doc, lex):
        scores = np.zeros(2)
        for token in doc:
            try:
                score = lex[token]
            except KeyError:
                score = 0
            if score>0:
                scores[0] += 1
            elif score <0:
                scores[1] += 1
        return scores
    
    def extract_binary_emos(self, doc, lex):
        scores = 0
        for token in doc:
            try:
                score = lex[token]
            except KeyError:
                score = 0
            if score>0:
                scores += 1
        return scores
    
    def extract_float(self, doc, lex):
        scores = 0.0
        for token in doc:
            try:
                score = lex[token]
            except KeyError:
                score = 0.0
            scores += score
        return scores    
    
    
    def read_AFINN(self):
        d = {}
        with open(self.folder+"AFINN-111.txt", encoding='utf-8') as f: #, encoding='utf-8'
            for line in f:
                tmp = line.split('\t')
                d[tmp[0]] = int(tmp[1])
        f.close()
        return d
    
    def read_general_inquirer(self):
        d = {}
        with open(self.folder+"GeneralInquirer.txt", encoding='utf-8') as f: #, encoding='utf-8'
            for line in f:
                tmp = line.split('\t')
                d[tmp[0]] = int(tmp[1][0:len(tmp[1])-3])
        f.close()
        return d
    
    def read_mpqa(self):
        d = {}
        with open(self.folder+"mpqa_05.txt", encoding='utf-8') as f: #, encoding='utf-8'
            for line in f:
                tmp = line.split(' ')
                key = tmp[2][6:]
                value = tmp[5]
                if value.find('positive') == -1:
                    value = -1
                else:
                    value = 1
                d[key] = value
        f.close()
        return d
    
    def read_opinion_finder(self):
        d = {}
        with open(self.folder+'positive-words.txt', encoding='utf-8') as f: #, encoding='utf-8'
            positive = f.read().splitlines()
        f.close()
        with open(self.folder+'negative-words.txt', encoding='ISO-8859-1') as f: #, encoding='ISO-8859-1'
            negative = f.read().splitlines()
        f.close()     
        for _ in positive:
            d[_] = 1
        for _ in negative:
            d[_] = -1
        return d
        
    def read_lexicon_unknown(self):
        d = {}
        with open(self.folder+"lexicon.txt", encoding='utf-8') as f: #, encoding='utf-8'
            for line in f:
                tmp = line.split(' ')
                d[tmp[0]] = float(tmp[1][0:len(tmp[1])-1])
        f.close()
        return d
    
    def read_unigram_pmi_hashtag(self):
        d = {}
        with open(self.folder+'unigrams-pmilexicon_hashtag.txt', encoding='utf-8') as f: #, encoding='utf-8'
            for line in f:
                tmp = line.split('\t')
                d[tmp[0]] = float(tmp[1])
        f.close()
        return d
    
    def read_unigram_pmi_sentiment140(self):
        d = {}
        with open(self.folder+'unigrams-pmilexicon_sentiment140.txt', encoding='utf-8') as f: #, encoding='utf-8'
            for line in f:
                tmp = line.split('\t')
                d[tmp[0]] = float(tmp[1])
        f.close()
        return d       
    
    def read_nrc_wordlevel_emotions(self):
        anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust = {},{},{},{},{},{},{},{},{},{}
        with open(self.folder+'NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', encoding='utf-8') as f: #, encoding='utf-8'
            for line in f:
                tmp = line.split('\t')
                key = tmp[0]
                emo = tmp[1]
                val = tmp[2]
                if emo=='anger':
                    anger[key] = int(val)
                elif emo=='anticipation':
                    anticipation[key] = int(val)
                elif emo=='disgust':
                    disgust[key] = int(val)
                elif emo=='fear':
                    fear[key] = int(val)
                elif emo=='joy':
                    joy[key] = int(val)
                elif emo=='negative':
                    negative[key] = int(val)
                elif emo=='positive':
                    positive[key] = int(val)
                elif emo=='sadness':
                    sadness[key] = int(val)
                elif emo=='surprise':
                    surprise[key] = int(val)
                elif emo=='trust':
                    trust[key] = int(val)
        f.close()
        return anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust
                
    def read_nrc_emotions(self):
        anticipation, fear, anger, trust, surprise, sadness, joy, disgust = {},{},{},{},{},{},{},{}
        with open(self.folder+'NRC-Hashtag-Emotion-Lexicon-v0.2.txt', encoding='utf-8') as f: #, encoding='utf-8'
            for line in f:
                tmp = line.split('\t')
                key = tmp[1]
                emo = tmp[0]
                val = tmp[2][0:len(tmp[2])-2]
                if emo=='anger':
                    anger[key] = float(val)
                elif emo=='anticipation':
                    anticipation[key] = float(val)
                elif emo=='disgust':
                    disgust[key] = float(val)
                elif emo=='fear':
                    fear[key] = float(val)
                elif emo=='joy':
                    joy[key] = float(val)
                elif emo=='sadness':
                    sadness[key] = float(val)
                elif emo=='surprise':
                    surprise[key] = float(val)
                elif emo=='trust':
                    trust[key] = float(val)
        f.close()
        return anticipation, fear, anger, trust, surprise, sadness, joy, disgust


# (1) Unzip the lexicon folder somewhere
# (2) Alter the self.folder bit in the first line of the __init__() to match the path to your local dir where the lexicons are saved
# (3) I think that “doc” should be a list of keywords here
# (4) Done; you now have the lexicon scores for the document (you can also try normalising them by number of tokens)

#lexicon = lexicon_reader() 
#lexicon.get_lexicon_scores(doc)