import numpy as np


def domain(gpos, x, dm, tnodes):
    dif = np.dot(gpos, np.ones((1, tnodes)))-x
    a = dm-abs(dif)
    b = np.vstack((a, np.zeros((1, tnodes))))
    c = np.all(b >= -100*np.finfo(np.float).eps, axis=0)

    v = np.where(c == 1)[0]

    di = gpos-x[:, v]

    return v, di
