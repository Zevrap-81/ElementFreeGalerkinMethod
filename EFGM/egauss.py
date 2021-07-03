'''
    taken as reference from another programming language

    Refer[6] in references
'''

import numpy as np
# This function returns the gauus points ,  their weights and jacobian


def egauss(xc, node, gauss, ncells):
    index = 0
    one = np.ones((1, 4))
    xcord = np.zeros((1, 4))
    ycord = np.zeros((1, 4))
    numq2 = ncells*4**2
    gs = np.zeros((4, numq2), dtype=np.float16)
    # print(gs.shape)
    sign1 = np.array([-1, +1, +1, -1])
    sign2 = np.array([-1, -1, +1, +1])
    l = gauss.shape[1]
    # print(l)
    for e in range(ncells):
        for j in range(4):
            GI = int(node[j, e])
#             print(GI)                                                # Global index number
            xcord[:, j] = xc[0, GI]
            # xcord.shape
            # print(xcord)                                                                    # x-cord of global index
            # y-cord of global index
            ycord[:, j] = xc[1, GI]
        for i in range(l):
            for j in range(l):
                eta1 = gauss[0, i]
                eta2 = gauss[0, j]
                # shape function in 2D
                N = 0.25*(one+eta2*sign1)*(one+eta1*sign2)
    #             print(N)
                dNeta2 = 0.25*sign1*(one+eta1*sign2)
                # print(dNeta1.shape)                                       # derivative of shape function w.r.t x-axis transformed(eta1)
                # derivative of shape function w.r.t y-axis transformed(eta2)
                dNeta1 = 0.25*sign2*(one+eta2*sign1)
               # calculation for the jacobian
                xeta2 = np.dot(dNeta2, np.transpose(xcord))
                # print(xeta1.shape)
                yeta2 = np.dot(dNeta2, np.transpose(ycord))
    #             print(yeta1)
                xeta1 = np.dot(dNeta1, np.transpose(xcord))
                # print(xeta2)
                yeta1 = np.dot(dNeta1, np.transpose(ycord))
                # print(yeta1)
                jacobian = np.dot(xeta2, yeta1)-np.dot(xeta1, yeta2)
                # print((xeta1*yeta2))
                # print(jacobian.shape)
                xq = np.dot(N, np.transpose(xcord))
                yq = np.dot(N, np.transpose(ycord))
                # gauus point for x-cord
                gs[0, index] = xq
                # gauss point for y-cord
                gs[1, index] = yq
                gs[2, index] = gauss[1, i]*gauss[1, j]           # gauss weight
                gs[3, index] = jacobian
                index = index+1

        # print(ycord)
    return gs
# print(gs.T)
# weight=gs[3,639]
# print(weight.shape)
# gs[0:2,639]
# gs[3,639]
# print(GI)
# gs.shape
