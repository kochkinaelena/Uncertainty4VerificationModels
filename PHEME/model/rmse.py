
import math

def rmse(Y_true, Y_pred, confidence):
    correct = 0
    errors = []
    
    for i in range(len(Y_true)):
        if Y_true[i]==Y_pred[i]:
            correct += 1
            errors.append( (1-confidence[i]) ** 2 )
        elif Y_true[i]=='unverified':
            errors.append( (confidence[i]) ** 2 )
        else:
            errors.append(1.0)   
       
    rmse = math.sqrt( sum(errors) / len(errors) )
    
    return rmse
