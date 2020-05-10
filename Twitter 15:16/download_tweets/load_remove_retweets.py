import os
import re
import pickle

def listdir_nohidden(path):
    files = os.listdir(path)
    newfiles = [i for i in files if i[0] != '.']
    return newfiles

savepath15 = "preprocessed_data/twitter15"
savepath16 = "preprocessed_data/twitter16"

path15 = "twitter15/tree"
path16 = "twitter16/tree"


def load_remove_rt_save(path, savepath):
    
    trees = listdir_nohidden(path)

    for t in trees:
    
        structure = []
        tweets = []
        lines = []
        fpath = os.path.join(path, t)
        
        for line in open(fpath):
            lines.append(line)
            line = re.sub("[^a-zA-Z0-9.]", " ", line)
            line = re.sub(' +',' ',line)
            line = line.rstrip()
            line = line.lstrip()
            listfromline = line.split(' ')
            
            assert len(listfromline)==6
            
            parentid = listfromline[1]
            childid = listfromline[4]
            
            if parentid=='ROOT':
                rootpath = os.path.join(savepath,childid)
                if not os.path.exists(rootpath):
                    os.mkdir(rootpath)
            
            if parentid == childid:
                continue
            else:
                structure.append([parentid, childid])
                if (parentid not in tweets) and parentid != 'ROOT':
                    tweets.append(parentid)
                if childid not in tweets:
                    tweets.append(childid)
        
        with open(os.path.join(rootpath, 'structure.pkl'), 'wb') as fp:
            pickle.dump(structure, fp)
            
        with open(os.path.join(rootpath, 'tweets.pkl'), 'wb') as fp:
            pickle.dump(tweets, fp)


load_remove_rt_save(path15, savepath15)
load_remove_rt_save(path16, savepath16)

