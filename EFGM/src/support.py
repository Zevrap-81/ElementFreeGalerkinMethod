import numpy as np

from EFGM.common.parameters import Parameters


def support(g_point, params:Parameters):
    dif = g_point- params.mesh.mesh.points[:, 0:params.mesh.num_dims]
    a = params.domain.dm-abs(dif)
    b = np.all(a >= -100*np.finfo(np.float).eps, axis=1) #accounting for floating point erors
    v = np.where(b == 1)[0]
    di = g_point-params.mesh.mesh.points[v, 0:params.mesh.num_dims]
    return v, di
