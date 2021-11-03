import numpy as np
import scipy
# SciPy Linear Algebra Library
from scipy.linalg import lu, solve_triangular

from EFGM.common.parameters import Parameters
from EFGM.src.gauss import GaussPoint
from .cubwgt import *
from .expwgt import *
import pytest
import matplotlib.pyplot as plt


def shape(g_point:GaussPoint, params: Parameters):

    gam = np.zeros((3, 3))
    ones = np.ones((g_point.len,1))
    nv = params.mesh.mesh.points[g_point.support_nodes, 0:params.mesh.num_dims]
    lin_base = np.hstack((ones, nv))  # polynomial basis(1,x,y)
    
    w, dwdx, dwdy = expwgt(g_point, params)

    # Uncomment to use cubic weight function
    # dif=g_ppg = np.insert(g_point*ones-nv
    # t=dm[0:2,v]/params.domain.dmax
    # w,dwdx,dwdy=cubwgt(dif,t,v,params.domain.dmax,dm)

    pp = np.einsum('xi, xo->xio', lin_base, lin_base) 
    A = (pp * w[:, None, None]).sum(axis=0)
    dAdx= (pp *dwdx[:, None, None]).sum(axis=0)
    dAdy= (pp *dwdy[:, None, None]).sum(axis=0)

    B = lin_base*w[:, None]
    dBdx = lin_base*dwdx[:, None]
    dBdy = lin_base*dwdy[:, None]
    B, dBdx, dBdy = B.T, dBdx.T, dBdy.T

    pg = np.insert(g_point.coord, 0, 1)

    P, L, U = lu(A)
    
    for i in range(3):             # 0 for phi, 1 for dphi/dx and 2 for dphi/dy
        if i == 0:
            C = P.dot(pg)
            gam[i] = solve_triangular(U,solve_triangular(L,C, lower=True))

        elif i == 1:
            C = P.dot(np.array([0, 1, 0])-dAdx.dot(gam[0]))
            gam[i] = solve_triangular(U, solve_triangular(L, C, lower=True))

        elif i == 2:
            C = P.dot(np.array([0, 0, 1])-dAdy.dot(gam[0]))
            gam[i] = solve_triangular(U, solve_triangular(L, C, lower=True))

    phi = np.dot(gam[0], B)
    dphix = np.dot(gam[1], B) + np.dot(gam[0], dBdx)
    dphiy = np.dot(gam[2], B) + np.dot(gam[0], dBdy)
    return phi, dphix, dphiy

