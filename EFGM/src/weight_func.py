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

def cubwgt(g_point:GaussPoint, params:Parameters):
    dm = params.domain.dm
    
    drdx=np.sign(g_point.dists[:,0])
    drdy=np.sign(g_point.dists[:,1])
    rx=np.abs(g_point.dists[:,0])/dm[g_point.support_nodes,0]
    ry=np.abs(g_point.dists[:,1])/dm[g_point.support_nodes,1]
    
    wx = np.empty(g_point.len)
    wy = np.empty(g_point.len)

    dwdx = np.empty(g_point.len)
    dwdy = np.empty(g_point.len)
    #print(drdx)
    id_x = rx>0.5 #all indices where rx >0.5
    wx[id_x]= 4/3 -4*rx[id_x] +4*rx[id_x]**2 -4/3*rx[id_x]**3
    dwdx[id_x]= (-4 +8*rx[id_x] -4*rx[id_x]**2)*drdx[id_x]
    # elif rx<=0.5:
    wx[~id_x]=2/3 -4*rx[~id_x]**2+ 4*rx[~id_x]**3
    dwdx[~id_x]=(-8*rx[~id_x] +12*rx[~id_x]**2)*drdx[~id_x]

    id_y = ry>0.5
    wy[id_y]= 4/3- 4*ry[id_y] +4*ry[id_y]**2 -4/3*ry[id_y]**3
    dwdy[id_y]=(-4 +8*ry[id_y] -4*ry[id_y]**2)*drdy[id_y]
    # elif ry<=0.5:
    wy[~id_y]=2/3- 4*ry[~id_y]**2+ 4*ry[~id_y]**3
    dwdy[~id_y]=(-8*ry[~id_y] +12*ry[~id_y]**2)*drdy[~id_y]

    w=wx*wy
    dwdx*=wy
    dwdy*=wx
    return (w, dwdx, dwdy)
