import numpy as np
def cubwgt(dif,t,v,dmax,dm):
    l=len(v)
    #print(l)
    w=np.zeros((1,l))
    dwdx=np.zeros((1,l))
    dwdy=np.zeros((1,l))
    for i in range(l):
        drdx=np.sign(dif[0,i])/dm[0,v[i]]
        drdy=np.sign(dif[1,i])/dm[1,v[i]]
        rx=abs(dif[0,i])/dm[0,v[i]]
        ry=abs(dif[1,i])/dm[1,v[i]]
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
        w[:,i]=wx*wy
        dwdx[:,i]=wy*dwx
        dwdy[:,i]=wx*dwy
    return (w,dwdx,dwdy)
#(w,dwdx,dwdy)=cubwgt(dif,t,v,dmax,dm)
#print(drdx)
