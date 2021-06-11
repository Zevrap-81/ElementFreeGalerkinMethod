import numpy as np


def expwgt(di, v, dmax, dm):
    l = len(v)
    w = np.zeros((1, l))
    dwdx = np.zeros((1, l))
    dwdy = np.zeros((1, l))
    c = dm[:, v]/dmax
    for i in range(l):
        didx = np.sign(di[0, i])
        didy = np.sign(di[1, i])
        dic = np.abs(di[:, i])/c[:, i]
        wx = (np.exp(-(dic[0])**2)-np.exp(-(dm[0, i]/c[0, i])
                                          ** 2))/(1-np.exp(-(dm[0, i]/c[0, i])**2))
        diwx = -2*(dic[0])*np.exp(-(dic[0])**2) / \
            (1-np.exp(-(dm[0, i]/c[0, i])**2))*didx
        wy = (np.exp(-(dic[1])**2)-np.exp(-(dm[1, i]/c[1, i])
                                          ** 2))/(1-np.exp(-(dm[1, i]/c[1, i])**2))
        diwy = -2*(dic[1])*np.exp(-(dic[1])**2) / \
            (1-np.exp(-(dm[1, i]/c[1, i])**2))*didy
        w[:, i] = wx*wy
        dwdx[:, i] = wy*diwx
        dwdy[:, i] = wx*diwy
    return (w, dwdx, dwdy)
