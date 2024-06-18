import numpy as np

def np_ndarray(a):
    if isinstance(a,np.ndarray):
        if len(a.shape)==2:
            return a
        return a.reshape(1,-1)
    else:
        if isinstance(a[0],list) or isinstance(a[0],tuple):
            return np.array(a,np.float32)
        else:
            return np.array([a],np.float32)
        

def np_toList(a):
    return list(map(lambda x:x.tolist()[0],a))