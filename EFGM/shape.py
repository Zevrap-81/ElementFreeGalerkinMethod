import numpy as np
import scipy
# SciPy Linear Algebra Library
import scipy.linalg
from cubwgt import *
from expwgt import *
import pytest
import matplotlib.pyplot as plt


def shape(gpos, dmax, x, v, dm, di):

    L = len(v)
    gam = np.zeros((3, 3))
    won = np.ones((1, L))
    nv = x[0:2, v]
    lin_p = np.vstack((won, nv))  # polynomial basis(1,x,y)

    w, dwdx, dwdy = expwgt(di, v, dmax, dm)

    # Uncomment to use cubic weight function
    # dif=gpos*won-nv
    # t=dm[0:2,v]/dmax
    # w,dwdx,dwdy=cubwgt(dif,t,v,dmax,dm)

    B = lin_p*np.vstack((w, w, w))

    pp = np.zeros((3, 3))
    aa = np.zeros((3, 3))
    daax = np.zeros((3, 3))
    daay = np.zeros((3, 3))

    for i in range(L):
        pp = np.outer(lin_p[:, i], lin_p[:, i])
        aa += w[0, i]*pp
        daax += dwdx[0, i]*pp
        daay += dwdy[0, i]*pp

    pg = np.insert(gpos, 0, 1)

    # P, L, U = scipy.linalg.lu(aa)
    P = np.eye(3)
    # LU Decomposition of aa matrix
    for i in range(3):             # 0 for phi, 1 for dphi/dx and 2 for dphi/dy
        if i == 0:
            C = P.dot(pg)
            gam[:, i] = np.linalg.solve(aa, C)

        elif i == 1:
            C = P.dot(np.array([0, 1, 0])-daax.dot(gam[0:3, 0]))
            gam[:, i] = np.linalg.solve(aa, C)

        elif i == 2:
            C = P.dot(np.array([0, 0, 1])-daay.dot(gam[0:3, 0]))
            gam[:, i] = np.linalg.solve(aa, C)

        # D0=C[0]
        # D1=(C[1]-L[1,0]*D0)
        # D2=(C[2]-L[2,0]*D0-L[2,1]*D1)
        # gam[2,i]=D2/U[2,2]
        # gam[1,i]=(D1-U[1,2]*gam[2,i])/(U[1,1])
        # gam[0,i]=(D0-U[0,1]*gam[1,i]-U[0,2]*gam[2,i])/U[0,0]

    # gam = np.linalg.solve(aa,pg)
    # print(gam)
    # exit()
    phi = np.dot(gam[0:3, 0], B)

    dbx = lin_p*np.vstack((dwdx, dwdx, dwdx))
    dby = lin_p*np.vstack((dwdy, dwdy, dwdy))
    dphix = np.dot(gam[0:3, 1], B) + np.dot(gam[0:3, 0], dbx)
    dphiy = np.dot(gam[0:3, 2], B) + np.dot(gam[0:3, 0], dby)
    return phi, dphix, dphiy

# dif= gpos-x[:,v]
# t=dm[:,v]/dmax
# w,dwdx,dwdy= cubwgt(dif,t,v,dmax,dm)
