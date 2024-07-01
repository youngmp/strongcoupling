
"""
library functions for right-hand sides
"""

import numpy as np
import matplotlib.pyplot as plt


def rhs_avg_2d(t,y,a,eps=0,del1=0,miter=None):
    """ for forcing only. 2d."""

    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    th,ps = y
    dth = 0
    dps = a.system1.kappa_val*ps
    
    for i in range(nn):
        dth += eps*ps**i*a.hz_lam[i](th)
        dps += eps*ps**i*a.hi_lam[i](th)
    dth -= del1/a._m[1]
    return np.array([dth*a._n[1],dps*a._m[1]])

def rhs_avg_4d(t,y,a,eps=0,del1=0,miter=None):
    """ for coupling only. 4d."""

    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    th0,ps0,th1,ps1 = y
    dth0=0;dth1=0
    dps0 = a.system1.kappa_val*ps0
    dps1 = a.system2.kappa_val*ps1
    
    for k in range(nn):
        dth0 += eps*ps0**k*a.hz_lam[k](th0)
        dps0 += eps*ps0**k*a.hi_lam[k](th0)
        
        dth1 += eps*ps1**k*a.hz_lam[k](th1)
        dps1 += eps*ps1**k*a.hi_lam[k](th1)
    dth -= del1/a._m[1]
    return np.array([dth*a._n[1],dps*a._m[1]])

def rhs_avg_1d(t,th,a,eps=0,del1=0,miter=None):
    """ for forcing only. 1d"""

    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    dth = 0
    
    for i in range(nn):
        dth += eps**(i+1)*a.system1.h['lam'][i](th)
    dth -= del1/a._m[1]
    return dth*a._n[1]


def rhs_avg_1dc(t,y,a,eps=0,del1=0,miter=None):
    """ for coupling only. 1d"""
    system1 = a.system1; system2 = a.system2
    
    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    h = 0
    for i in range(nn):
        het = 4*del1*np.mean(system1.z['dat'][i])/a._n[1]
        h += eps**(i+1)*(het + system1.h['lam'][i](y) - system2.h['lam'][i](y))
    return h*a._n[1]
    
    


def _redu(t,y,a,eps=.01,del1=0):
    th,ps,tf = y
    
    u = a.system1.forcing_fn(tf)
    
    dth = 1 + eps*a.th_lam(th,ps)*u
    dps = a.system1.kappa_val*ps + eps*a.ps_lam(th,ps)*u
    dtf = 1 + del1/a._m[1]
    return np.array([dth*a._n[1],dps*a._n[1],dtf*a._m[1]])


def _redu_moving(t,y,a,eps=.01,del1=0):
    th,ps,tf = y

    tf_m = tf+a._m[1]*t
    th_m = th+a._n[1]*t
    
    u = a.system1(tf_m)
    
    dth = eps*a.th_lam(th_m,ps)*u
    dps = a.system1.kappa_val*ps + eps*a.ps_lam(th_m,ps)*u
    dtf = a.om*del1/a._m[1]
    return np.array([dth*a._n[1],dps*a._n[1],dtf*a._m[1]])


def _redu_moving_avg(t,y,a,eps=.01,del1=0):

    s,ds = np.linspace(0,2*np.pi*a._m[1],a.NH,retstep=True)
    th,ps,tf = y

    om = a._n[1]/a._m[1]
    in1 = th-om*tf+om*s;
    in2 = s

    z = a.th_lam(in1,ps)
    i = a.ps_lam(in1,ps)
    u = a.system2(in2)
    
    dth = eps*np.sum(z*u)*ds/(2*np.pi*a._m[1])
    dps = a.system1.kappa_val*ps + eps*np.sum(i*u)*ds/(2*np.pi*a._m[1])
    dtf = del1
    return np.array([dth*a._n[1],dps*a._n[1],dtf])
