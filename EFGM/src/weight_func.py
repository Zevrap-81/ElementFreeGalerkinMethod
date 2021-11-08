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
    #print(g_point.len)
    w=np.empty(g_point.len)
    dwdx=np.empty(g_point.len)
    dwdy=np.empty(g_point.len)

    for i in range(g_point.len):
        drdx=np.sign(g_point.dists[i,0])/dm[g_point.support_nodes[i],0]
        drdy=np.sign(g_point.dists[i,1])/dm[g_point.support_nodes[i],1]
        rx=abs(g_point.dists[i,0])/dm[g_point.support_nodes[i],0]
        ry=abs(g_point.dists[i,1])/dm[g_point.support_nodes[i],1]
        #print(drdx)
        if rx>0.5:
            wx=(4/3)-(4*rx)+(4*rx*rx)-(4/3)*(rx**3)
            dwx=(-4+(8*rx)-(4*rx**2))*drdx
        elif rx<=0.5:
            wx=(2/3)-(4*rx*rx)+(4*rx**3)
            dwx=((-8*rx)+(12*rx**2))*drdx
        if ry>0.5:
            wy=(4/3)-(4*ry)+(4*ry*ry)-(4/3)*(ry**3)
            dwy=(-4+(8*ry)-(4*ry**2))*drdy
        elif ry<=0.5:
            wy=(2/3)-(4*ry*ry)+(4*ry**3)
            dwy=((-8*ry)+(12*ry**2))*drdy
        #print(i)
        w[i]=wx*wy
        dwdx[i]=wy*dwx
        dwdy[i]=wx*dwy
    return (w,dwdx,dwdy)
