import numpy as np


def nodes_domain(dmax, leng, hgt, ndivl, ndivw, tnodes):

    dm = np.zeros((2, tnodes), dtype=np.float16)
    hx = leng/ndivl
    hy = hgt/ndivw
    dm[0] = dmax*hx*np.ones(tnodes, dtype=np.float16)
    dm[1] = dmax*hy*np.ones(tnodes, dtype=np.float16)
    return dm
