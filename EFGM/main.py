import scipy
# SciPy Linear Algebra Library
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
import pytest
from meshgeneration import *
from pgauss import *
from egauss import *
from nodes_domain import nodes_domain
from domain import domain
from shape import shape


# total no.of divisions along x-axis
ndivl = 24
# total no. of division along y-axis
ndivw = 12
leng = 48
hgt = 12
P = 1000  # load
E = 30e6
poi = 0.3
Dmat = np.dot((E/(1-poi**2)),
              np.array([[1, poi, 0], [poi, 1, 0], [0, 0, (1-poi)/2]]))
Cmat = np.array([[1, -poi, 0], [-poi, 1, 0], [0, 0, 1+poi]])/E

# Nodes generation
x, _, _, tnodes = meshgeneration(leng, hgt, ndivl, ndivw)  # nodal mesh
plt.scatter(x[0, :], x[1, :])
plt.show()


# Determination of domain of influence
dmax = 7
dm = nodes_domain(dmax, leng, hgt, ndivl, ndivw, tnodes)

ndivlq = 24             # quadrature points
ndivwq = 12
xc, node, ncells, tcnodes = meshgeneration(
    leng, hgt, ndivlq, ndivwq)  # gauss points mesh


quado = 4       # order of quadrature
gauss = pgauss()
numq2 = ncells*quado**2
gs = np.zeros((4, numq2), dtype=np.float16)
gs = egauss(xc, node, gauss, ncells)


# Loop over gauss points to assemble discrete equations
k = np.zeros((tnodes*2, tnodes*2))
flag = 0
for gg in gs.transpose():
    gpos = np.array([gg[0:2]]).transpose()
    weight = gg[2]
    jac = gg[3]
    v, di = domain(gpos, x, dm, tnodes)
    L = len(v)
    en = np.zeros((2*L), dtype=np.int16)

    phi, dphix, dphiy = shape(gpos, dmax, x, v, dm, di)
    assert pytest.approx(sum(phi), 10**(-3)) == 1  # test 1
    # print('Success: the sum of shape functions at any point is 1')

    assert pytest.approx(sum(phi*x[0, v]), 10**(-4)) == gpos[0, 0]
    # print('Success')

    Bmat = np.zeros((3, 2*L), dtype=np.float16)
    for j in range(L):
        Bmat[0:3, (2*j):2*j+2] = np.array([[dphix[j], 0],
                                           [0, dphiy[j]], [dphiy[j], dphix[j]]])

    for i in range(L):
        en[2*i] = 2*v[i]
        en[2*i+1] = 2*v[i]+1

    xx, yy = np.meshgrid(en, en)

    k[xx, yy] += ((weight*jac)*np.dot(np.transpose(Bmat), np.dot(Dmat, Bmat)))
    flag += 1

# setting up Boundary conditions
'''
    taken as reference from another programming language

    Refer[6] in references
'''
ind1 = 0
ind2 = 0
nnu = np.zeros((2, ndivw+1))
nt = np.zeros((2, ndivw+1))
for j in range(tnodes):
    if (x[0, j] == 0.0):
        nnu[0, ind1] = x[0, j]
        nnu[1, ind1] = x[1, j]
        ind1 = ind1+1
    if (x[0, j] == leng):
        nt[0, ind2] = x[0, j]
        nt[1, ind2] = x[1, j]
        ind2 = ind2+1
lthu = len(nnu[0])
ltht = len(nt[0])
ubar = np.zeros((lthu*2, 1))
f = np.zeros((tnodes*2, 1))

'''
    few lines are inspired from a code in another programming language
    refer[6]
'''

# Gauss points along the traction boundary
ind = 0
gauss = pgauss()
gst = np.zeros((4, (ltht-1)*4))
mark = np.zeros((1, 4))
for i in range(ltht-1):
    ycen = (nt[1, i+1]+nt[1, i])/2
    jcob = abs((nt[1, i+1]-nt[1, i])/2)
    for j in range(4):  # quado
        mark[0, j] = ycen-gauss[0, j]*jcob
        gst[0, ind] = nt[0, i]
        gst[1, ind] = mark[0, j]
        gst[2, ind] = gauss[1, j]
        gst[3, ind] = jcob
        ind = ind+1

# setting up Gauss points along the displacement boundary

# integrating forces along the boundary
Imo = (1/12)*(hgt**3)
for gt in gst.transpose():
    gpos = np.array([gt[0:2]]).transpose()
    weight = gt[2]
    jac = gt[3]
    v, di = domain(gpos, x, dm, tnodes)
    L = len(v)
    en = np.zeros((1, 2*L))
    force = np.zeros((1, 2*L))
    phi, dphix, dphiy = shape(gpos, dmax, x, v, dm, di)
    tx = 0
    ty = -(P/(2*Imo))*((hgt**2)/4-gpos[1, 0]**2)
    for i in range(L):
        en[:, 2*i] = (2*v[i])
        en[:, 2*i+1] = (2*v[i]+1)
        force[:, 2*i] = tx*phi[i]
        force[:, 2*i+1] = ty*phi[i]
        en = en.astype(int)
    f[(en)] = f[(en)]+jac*weight*np.transpose(force)

# setting up Gauss points along the displacement boundary
gsu = gst.copy()
gsu[0] = 0
qk = np.zeros((1, 2*lthu))


# integrate G matrix and Q vector along the displacement boundary
GG = np.zeros((tnodes*2, lthu*2))
ind1 = 0
ind2 = 0
for i in range(lthu-1):
    m1 = ind1
    m2 = m1+1
    y1 = nnu[1, m1]
    y2 = nnu[1, m2]
    le = y1-y2
    ind1 = ind1+1
    for j in range(quado):
        gpos = np.array([gsu[0:2, ind2]]).transpose()
        weight = gsu[2, j]
        jac = gsu[3, j]
        fac11 = (-P*nnu[1, m1])/(6*E*Imo)
        fac2 = P/(6*E*Imo)
        xp1 = gsu[0, ind2]
        yp1 = gsu[1, ind2]

        uxex1 = (6*leng-3*xp1)*xp1 + (2+poi)*(yp1**2 - (hgt/2)**2)
        uxex1 = uxex1*fac11
        uyex1 = 3*poi*(yp1**2)*(leng-xp1)+0.25*(4+5*poi) * \
            xp1*(hgt**2)+(3*leng-xp1)*(xp1**2)
        uyex1 = uyex1*fac2

        N1 = (gpos[1]-y2).item()/le
        N2 = 1-N1
        qk[:, 2*m1] = qk[:, 2*m1]-weight*jac*N1*uxex1
        qk[:, 2*m1+1] = qk[:, 2*m1+1]-weight*jac*N1*uyex1
        qk[:, 2*m2] = qk[:, 2*m2]-weight*jac*N2*uxex1
        qk[:, 2*m2+1] = qk[:, 2*m2+1]-weight*jac*N2*uyex1
        v, di = domain(gpos, x, dm, tnodes)

        phi, dphix, dphiy = shape(gpos, dmax, x, v, dm, di)
        L = len(v)
        ind2 = ind2+1
        for n in range(L):
            G1 = -weight*jac*phi[n]*np.array([[N1, 0], [0, N1]])
            G2 = -weight*jac*phi[n]*np.array([[N2, 0], [0, N2]])
            c1 = 2*v[n]
            c2 = 2*v[n]+2
            c3 = 2*m1
            c4 = 2*m1+2
            c5 = 2*m2
            c6 = 2*m2+2
            xx1, yy1 = np.meshgrid(np.arange(c1, c2), np.arange(c3, c4))
            xx2, yy2 = np.meshgrid(np.arange(c1, c2), np.arange(c5, c6))
            GG[xx1, yy1] = GG[xx1, yy1]+G1
            GG[xx2, yy2] = GG[xx2, yy2]+G2


# Boundary conditions using Lagrange multiplier
'''
    taken as reference from another programming language

    Refer[6] in references
'''
f = np.vstack((f, np.zeros((lthu*2, 1))))
d = np.zeros((tnodes*2+2*lthu, 1))
f[tnodes*2:tnodes*2+2*lthu] = np.transpose(-qk)
a = (np.hstack((k, GG)))
b = (np.hstack((np.transpose(GG), np.zeros((lthu*2, lthu*2)))))
m = np.vstack((a, b))

d = np.linalg.solve(m, f)

u = d[0:2*tnodes]
u2 = np.zeros((2, tnodes))
for i in range(tnodes):
    u2[0, i] = u[2*i]
    u2[1, i] = u[2*i+1]


# solving for output variables

displ = np.zeros((1, 2*tnodes))
ind = 0
for gg in x.transpose():
    gpos = np.array([gg[0:2]]).transpose()
    v, di = domain(gpos, x, dm, tnodes)
    phi, dphix, dphiy = shape(gpos, dmax, x, v, dm, di)
    displ[:, 2*ind] = np.dot(phi, np.transpose(u2[0, v]))
    displ[:, 2*ind+1] = np.dot(phi, np.transpose(u2[1, v]))
    ind = ind+1


ind = 0
stress = np.zeros((3, tnodes), dtype=np.float16)
stressex = np.zeros((3, tnodes), dtype=np.float16)
for i in range(tnodes):
    gpos[0] = x[0, i]
    gpos[1] = x[1, i]
    v, di = domain(gpos, x, dm, tnodes)
    L = len(v)
    en = np.zeros((2*L), dtype='int')

    phi, dphix, dphiy = shape(gpos, dmax, x, v, dm, di)
    Bmat = np.zeros((3, 2*L))
    for j in range(L):
        Bmat[0:3, (2*j):2*j+2] = np.array([[dphix[j], 0],
                                           [0, dphiy[j]], [dphiy[j], dphix[j]]])
    for I in range(L):
        en[2*I] = (2*v[I])
        en[2*I+1] = (2*v[I]+1)

    stress[:, i] = (Dmat@Bmat@u[en]).flatten()
    stressex[0, i] = (1/Imo)*P*(leng-gpos[0, 0])*gpos[1, 0]
    stressex[1, i] = 0
    stressex[2, i] = -0.5*(P/Imo)*(0.25*(hgt**2) - gpos[1, 0]**2)


fig, ax = plt.subplots()

# surf=ax.plot_trisurf(x[0], x[1], stress[2])

# ax.plot(x,hypothesis,"--",color="black",label="true function")
# ax.plot(x,y_observed,"o",color="blue",label="noisy data points")
ax.plot(x[1, 156:169], stress[2, 156:169], color='b', label='Numerical')
ax.plot(x[1, 156:169], stressex[2, 156:169], color='r', label='Analytical')
ax.legend()
plt.xlabel("nodes along width")
plt.ylabel("τ (psi)")
plt.title("shear stress along cross-section")
plt.show()
# Assuming plane strain
# No infuence of crack tip only the length?
# The final formula removes the influence of E....
# crack = np.array([[0,1],[11,1]])
# a= np.linalg.norm(crack[:-1]-crack[1:], axis=1).sum()
# # Taking some other value of a thats why again calculated here
# K = j_integral(a,hgt,P,E,poi)
# print(K)
