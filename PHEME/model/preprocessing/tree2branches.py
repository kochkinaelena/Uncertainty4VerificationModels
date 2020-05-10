from copy import deepcopy

def tree2branches(root):
    node = root
    parent_tracker = []
    parent_tracker.append(root)
    branch = []
    branches = []
    i=0
    siblings = None
    while True:
        node_name = list(node.keys())[i]
        #print node_name
        branch.append(node_name)
        # get children of the node
        first_child = list(node.values())[i] # actually all chldren, all tree left under this node
        if first_child != []: # if node has children
            node = first_child      # walk down
            parent_tracker.append(node)
            siblings = list(first_child.keys())
            i=0 # index of a current node
        else:
            branches.append(deepcopy(branch))
            if siblings != None:
                i=siblings.index(node_name) # index of a current node
                while i+1>=len(siblings): # if the node does not have next siblings
                    if node is parent_tracker[0]: # if it is a root node
                        return branches
                    del parent_tracker[-1]
                    del branch[-1]
                    node = parent_tracker[-1]      # walk up ... one step
                    node_name = branch[-1]
                    siblings = list(node.keys())
                    i=siblings.index(node_name)
                i=i+1    # ... walk right
    #            node =  parent_tracker[-1].values()[i]
                del branch[-1]
            else:
                return branches
            
            
def tree2timeline(conversation):
    timeline = []
#    given x and y sort x according to y values
#    [x for (y,x) in sorted(zip(Y,X))] - order matters here
    
    timeline.append(conversation['source']['id_str'])
    replies = conversation['replies']
    replies_idstr = []
    replies_timestamp = []
    for reply in replies:
        replies_idstr.append(reply['id_str'])
        replies_timestamp.append(reply['created_at'])
    
    sorted_replies = [x for (y,x) in sorted(zip(replies_timestamp,replies_idstr))]
    timeline.extend(sorted_replies)
    return timeline
