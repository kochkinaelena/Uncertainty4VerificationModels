from copy import deepcopy
import numpy as np
import datetime

def tree2branches(struct, conversation):
    
    branches = []
    
    temp_struct = []
    
    for i in struct:
        if i not in temp_struct:
            temp_struct.append(i)
    
    
    for i,b in enumerate(temp_struct):
        if b[0]=='ROOT':
            parent = b[1]
            del temp_struct[i]
            break
    
    
    all_rep_ids = [rep['id_str'] for rep in conversation['replies']]
    
    idx2del = []
    for i,b in enumerate(temp_struct):
        j = [b[1],b[0]]
        if j in temp_struct:
            idx = temp_struct.index(j)
            
            b0_children = []
            b1_children = []
            for temp_b in temp_struct:
                if (temp_b!=b) and (temp_b!=j):
                    if temp_b[0]==b[0]:
                        b0_children.append(temp_b[1])
                    if temp_b[0]==b[1]:
                        b1_children.append(temp_b[1])
            
            
            if b[0]==parent:
                idx2del.append(idx)
            else:
#                for k in temp_struct:
#                    if (k[1]==b[0]) and (k[0]!=b[1]):
#                        idx2del.append(idx)
#                    if (k[1]==b[1]) and (k[0]!=b[0]):
#                        idx2del.append(i)
                    
                        
                if (b[0] in all_rep_ids) and (b[1] in all_rep_ids):
                    for rep in conversation['replies']:
                        if rep['id_str']==b[0]:
                            t0 = datetime.datetime.strptime(rep['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                        if rep['id_str']==b[1]:            
                            t1 = datetime.datetime.strptime(rep['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                    if t0>t1:
                        idx2del.append(i)
                    elif t1>t0:
                        idx2del.append(idx)
                        
            
#            
                
                if (b[0] not in all_rep_ids) and (b[1] in all_rep_ids):
#                    print ("(b[0] not in all_rep_ids) and (b[1] in all_rep_ids) ")
                    if b0_children==[]:
#                        print ("empty path b0")
                        idx2del.append(i)
                if (b[0] in all_rep_ids) and (b[1] not in all_rep_ids):
#                    print ("(b[0] in all_rep_ids) and (b[1] not in all_rep_ids) ")
                    if b1_children==[]:
#                        print ("empty path b1")
                        idx2del.append(idx)
                if (b[0] not in all_rep_ids) and (b[1] not in all_rep_ids):
#                    print ("(b[0] not in all_rep_ids) and (b[1] not in all_rep_ids) ")
                    if b0_children==[] and b1_children==[]:
#                        print ("empty path")
                        idx2del.append(i)
                        idx2del.append(idx)
                        
#                        check if they are in rep_id
    
    
    for i,b in reversed(list(enumerate(temp_struct))):
        if i in idx2del:
            del temp_struct[i]
             

    while temp_struct!=[]:
        all_children = []
        for i,b in reversed(list(enumerate(temp_struct))):
            if b[0]==parent:
                all_children.append(b)
                del temp_struct[i]
        if branches == []:
            branches = all_children
        else:
            for ib, branch in reversed(list(enumerate(branches))):
                if branch[-1]==parent:
                    temp1 = deepcopy(branch)
                    temp2 = deepcopy(branch)
                    del branches[ib]
                    for child in all_children:
                        temp1.append(child[1])
                        branches.append(temp1)
                        temp1 = deepcopy(temp2)
        if temp_struct!=[]:
            parent = temp_struct[0][0]
            
    for b in branches:
        if b[-1] in b[:-1]:
            del b[-1]
            
    temp_branches = []
    
    for i in branches:
        if i not in temp_branches:
            temp_branches.append(i)
        
    branches = temp_branches
    
    return branches
