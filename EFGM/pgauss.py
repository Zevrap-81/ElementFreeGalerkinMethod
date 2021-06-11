import numpy as np
# This function returns a matrix with 4 gauss points and their weights
def pgauss():
    v=np.zeros((2,4))
    v[0,0] =-.861136311594052575224
    v[0,1] =-.339981043584856264803
    v[0,2] = -v[0,1]
    v[0,3] = -v[0,0]
    v[1,0] =.347854845137453857373
    v[1,1] =.652145154862546142627
    v[1,2] = v[1,1]
    v[1,3] = v[1,0]
    return v
v=pgauss()