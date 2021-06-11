import numpy as np
# A 2D meshgeneartion of the geometry


def meshgeneration(leng, hgt, ndivl, ndivw, *args, **kwargs):

    # total number of nodes
    tnodes = (ndivl+1)*(ndivw+1)
    # total number of cells
    ncells = ndivw*ndivl
    x = np.zeros((2, tnodes), dtype=np.float16)
    for i in range(ndivl+1):
        for j in range(ndivw+1):
            # x-cord of each node
            x[0, (ndivw+1)*i + j] = (leng/ndivl) * i
            # y-cord of each node
            x[1, (ndivw+1)*i + j] = -(hgt/ndivw) * j + hgt/2

    # connectivity of each cell w.r.t to the nodes
    node = np.zeros((4, ncells), dtype=np.int16)
    for j in range(ndivl):
        for i in range(ndivw):
            cell_no = j*ndivw+i
            node[0, cell_no] = cell_no+j
            node[1, cell_no] = node[0, cell_no]+1
            node[2, cell_no] = node[1, cell_no]+ndivw+1
            node[3, cell_no] = node[2, cell_no]-1
    return (x, node, ncells, tnodes)
