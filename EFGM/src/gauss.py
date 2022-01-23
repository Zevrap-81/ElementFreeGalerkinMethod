from dataclasses import dataclass

import numpy as np
from EFGM.common.parameters import Parameters
from EFGM.src.support import support

@dataclass
class GaussPoint:
    coord : np.ndarray = None
    weight : float = None
    jac_det : float = None
    support_nodes : np.ndarray = None
    dists : np.ndarray = None

    @property
    def len(self):
        return len(self.support_nodes)
        
def get_gauss(quad, params: Parameters):
    if len(quad)==params.mesh.num_dims:
        point= quad
        weight, jac_det= 0.0, 0.0 

    else:
        point= quad[0:params.mesh.num_dims]
        weight= quad[-2]
        jac_det= quad[-1]

    v, di = support(point, params)
    return GaussPoint(point, weight, jac_det, v, di)
    