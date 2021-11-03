import numpy as np
from EFGM.common.parameters import Parameters

from EFGM.src.gauss import GaussPoint


def expwgt(g_point: GaussPoint, params: Parameters):
    
    w = np.empty(g_point.len)
    dwdx = np.empty(g_point.len)
    dwdy = np.empty(g_point.len)
    c = params.domain.dm[g_point.support_nodes, :]/params.domain.dmax
    for i in range(g_point.len):
        didx = np.sign(g_point.dists[i,0])
        didy = np.sign(g_point.dists[i,1])
        dic = np.abs(g_point.dists[i])/c[i]
        wx = (np.exp(-(dic[0])**2)-np.exp(-(params.domain.dm[i,0]/c[i,0])
                                          ** 2))/(1-np.exp(-(params.domain.dm[i,0]/c[i,0])**2))
        diwx = -2*(dic[0])*np.exp(-(dic[0])**2) / \
            (1-np.exp(-(params.domain.dm[i,0]/c[i,0])**2))*didx
        wy = (np.exp(-(dic[1])**2)-np.exp(-(params.domain.dm[i,1]/c[i,1])
                                          ** 2))/(1-np.exp(-(params.domain.dm[i,1]/c[i,1])**2))
        diwy = -2*(dic[1])*np.exp(-(dic[1])**2) / \
            (1-np.exp(-(params.domain.dm[i,1]/c[i,1])**2))*didy
        w[i] = wx*wy
        dwdx[i] = wy*diwx
        dwdy[i] = wx*diwy
    return (w, dwdx, dwdy)
