import numpy as np 
import pytest
from meshgeneration import *
from nodes_domain import *
from domain import domain
from expwgt import expwgt
from cubwgt import cubwgt
import matplotlib.pyplot as plt 

def angle(vA, vB, len_vA):
    len_vB= np.linalg.norm(vB)
    theta= np.dot(vB.transpose(),vA)/(len_vA*len_vB)
    theta= np.arccos(theta)
    theta[np.where(np.isnan(theta))]=0              #if zero length
    return theta 
def polynomial_basis(nv, r, theta):
    p1= np.ones_like(r)
    p2= nv[0]
    p3= nv[1]
    R=np.sqrt(r)
    p4=R*np.cos(theta/2)            #first enriched polynomial basis                               
    p5=R*np.sin(theta/2)            #second enriched polynomial basis                                 
    p6=p5*np.sin(theta)             #third enriched polynomial basis
    p7=p4*np.sin(theta)             #fourth enriched polynomial basis
    enr_p=np.vstack((p1,p2,p3,p4,p5,p6,p7)) 
    return enr_p

def enrichedShape(gpos,dmax,x,v,dm,di, crack):
    L=len(v)
    gam=np.zeros((7,3))
    won=np.ones((1,L))
    nv=x[0:2,v]
    crack_tip= crack[-1]
    cc= (crack[-1]-crack[-2]).reshape(2,1)
    xic= nv-crack_tip.reshape(2,1)
    global r, theta
    r= np.linalg.norm(xic, axis=0, keepdims=True)
    theta= angle(xic,cc,r)       
    enr_p= polynomial_basis(nv, r, theta)

    dif=gpos*won-nv
    t=dm[0:2,v]/dmax
    w,dwdx,dwdy=expwgt(di,v,dmax,dm)
    B= enr_p*w
    aa=np.zeros((7,7))
    daax=np.zeros((7,7))
    daay=np.zeros((7,7))
    
    for i in range(L):
        pp=np.outer(enr_p[:,i],enr_p[:,i])
        aa+= w[0,i]*pp
        daax+=dwdx[0,i]*pp
        daay+=dwdy[0,i]*pp
    xc= gpos-crack_tip.reshape(2,1)
    r= np.linalg.norm(xc, axis=0, keepdims=True)
    theta= angle(xc,cc,r)       
    pg=polynomial_basis(gpos,r, theta)

    P=np.eye(7)
  
    for i in range (3):                 # 0 for phi, 1 for dphi/dx and 2 for dphi/dy
        if i==0:
            C= P.dot(pg.flatten()).astype(np.float64)
            gam[:,i] = np.linalg.solve(aa,C)

             
        elif i==1:
            den= 2*r**1.5 
            dp4x = (xc[0]*np.cos(theta/2) + xc[1]*np.sin(theta/2))/den
            dp5x = (xc[0]*np.sin(theta/2) - xc[1]*np.cos(theta/2))/den
            dp6x = (dp5x*np.sin(theta) - 2*xc[1]*np.sin(theta/2)*np.cos(theta))/den
            dp7x = (dp4x*np.sin(theta) - 2*xc[1]*np.cos(theta/2)*np.cos(theta))/den

            dpx = np.array([0,1,0,dp4x,dp5x,dp6x,dp7x],dtype=np.float16)
            C=P.dot(dpx-daax.dot(gam[:,0])).astype(np.float64)

            gam[:,i] = np.linalg.solve(aa,C)
            

        elif i==2:
            den= 2*r**1.5 
            dp4y = (xc[1]*np.cos(theta/2) - xc[0]*np.sin(theta/2))/den
            dp5y = (xc[1]*np.sin(theta/2) + xc[0]*np.cos(theta/2))/den
            dp6y = (dp5x*np.sin(theta) + 2*xc[0]*np.sin(theta/2)*np.cos(theta))/den
            dp7y = (dp4x*np.sin(theta) + 2*xc[0]*np.cos(theta/2)*np.cos(theta))/den
            dpy = np.array([0,0,1,dp4y,dp5y,dp6y,dp7y], dtype=np.float16)

            C=P.dot(dpy - daay.dot(gam[:,0])).astype(np.float64)
            gam[:,i] = np.linalg.solve(aa,C)
           

    phi=np.dot(gam[:,0],B)
    dbx=enr_p*dwdx
    dby=enr_p*dwdy
    dphix=np.dot(gam[:,1],B)+ np.dot(gam[:,0],dbx)
    dphiy=np.dot(gam[:,2],B) + np.dot(gam[:,0],dby)
    return phi,dphix,dphiy


if __name__=="__main__":
    '''
        Test case for radius and angle
    '''

    
    crack= np.array([[-1,0],[0,0]])
    cc= (crack[-1]-crack[-2]).reshape(2,1)
    c= crack[-1]

    x=np.array([[1,1]]).reshape(2,1)

    xc= x-c.reshape(2,1)
    r= np.linalg.norm(xc, axis=0, keepdims=True)

    theta= angle(xc,cc,r)
    assert pytest.approx(np.rad2deg(theta), 10**-3)==np.rad2deg(np.pi/4)
    print('Success')


    # leng, hgt, ndivl, ndivw, is_crack, n= 3,3,6,6,True,4
    # x,_,_,tnodes= meshgeneration(leng, hgt, ndivl, ndivw, is_crack, crack, n)
    
    # dmax=3.5
    # dm= nodes_domain(dmax, leng, hgt, ndivl, ndivw, tnodes, is_crack,n)
    # v,di= domain(gpos,x,dm,tnodes,is_crack, crack)
    # print('domain',v)
    # enrichedShape(gpos,dmax,x,v,dm,di,crack)
    # print('radius',r)
    # print('theta', np.rad2deg(theta))  
    
    # plt.scatter(x[0,:],x[1,:])
    # plt.plot(crack[:,0], crack[:,1], color='r')
    # plt.show() 