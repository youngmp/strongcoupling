"""
Example: Goodwin circadian oscillator from Gonze et al Biophys J 2005 

"""

# user-defined
#from nmCoupling import nmCoupling as nm
from StrongCoupling import StrongCoupling2
from response import Response as rsp

import numpy as np
import sympy as sym
from sympy import Matrix
import scipy as sp

import matplotlib.pyplot as plt

import os
import argparse

def rhs(t,z,pdict,option='value',idx=''):
    """
    Right-hand side of the Goodwin oscillator from
    Gonze et al Biophys J 2005
    
    Parameters
        
        t : float or sympy object.
            time
        z : array or list of floats or sympy objects.
            state variables of the thalamic model x,y,z,v
        pdict : dict of flots or sympy objects.
            parameter dictionary pdict[key], val. key is always a string
            of the parameter. val is either the parameter value (float) or 
            the symbolic version of the parameter key.
        option : string.
            Set to 'val' when inputs, t, z, pdict are floats. Set to
            'sym' when inputs t, z, pdict are sympy objects. The default
            is 'val'.
        
    Returns
        
        numpy array or sympy Matrix
            returns numpy array if option == 'val'
            returns sympy Matrix if option == 'sym'
            
    """
    
    x,y,z,v = z
    idx = str(idx)
    
    p = pdict
    n = p['n'+idx]
    om = pdict['om_fix'+idx]
    
    dx = p['v1'+idx]*p['k1'+idx]**n/(p['k1'+idx]**n+z**n)\
        - p['v2'+idx]*x/(p['k2'+idx]+x) + p['L'+idx]
    dy = p['k3'+idx]*x - p['v4'+idx]*y/(p['k4'+idx]+y)
    dz = p['k5'+idx]*y - p['v6'+idx]*z/(p['k6'+idx]+z)
    dv = p['k7'+idx]*x - p['v8'+idx]*v/(p['k8'+idx]+v)
    
    if option in ['value','val']:
        return om*np.array([dx,dy,dz,dv])
    elif option in ['sym','symbolic']:
        return om*Matrix([dx,dy,dz,dv])

def coupling(vars_pair,pdict,option='value',idx=''):
    """
    
    Ccoupling function between Goodwin oscillators
    
    E.g.,this Python function is the function $G(x_i,x_j)$
    in the equation
    $\\frac{dx_i}{dt} = F(x_i) + \\varepsilon G(x_i,x_j)$
    
    Parameters
    
        vars_pair : list or array
            contains state variables from oscillator A and B, e.g.,
            x1,y1,z1,v1,x2,y2,z2,v2
        pdict : dict of flots or sympy objects.
            parameter dictionary pdict[key], val. key is always a string
            of the parameter. val is either the parameter value (float) or 
            the symbolic version of the parameter key.
        option : string.
            Set to 'val' when inputs, t, z, pdict are floats. Set to
            'sym' when inputs t, z, pdict are sympy objects. The default
            is 'val'.
    
    Returns
        
        * numpy array or sympy Matrix
            * returns numpy array if option == 'val'. 
            returns sympy Matrix if option == 'sym'
    
    """
    x1,y1,z1,v1,x2,y2,z2,v2 = vars_pair
    idx = str(idx)
    
    K = pdict['K'+idx]
    #vc = pdict['eps']
    kc = pdict['kc'+idx]
    F = (v1+v2)/2

    om = pdict['om_fix'+idx]
    
    if option in ['value','val']:
        return om*np.array([K*F/(kc+K*F),0,0,0])
    elif option in ['sym','symbolic']:
        return om*Matrix([K*F/(kc+K*F),0,0,0])

def main():

    pd1 = {'v1':.84,'v2':.42,'v4':.35,'v6':.35,'v8':1,
           'k1':1,'k2':1,'k3':.7,'k4':1,'k5':.7,
           'k6':1,'k7':.35,'k8':1,'K':0.5,'kc':1,
           'n':6,'L':0,'eps':0,'om':1,'om_fix':1}

    kws1 = {'var_names':['x','y','z','v'],
            'pardict':pd1,
            'rhs':rhs,
            'coupling':coupling,
            'init':np.array([.3882,.523,1.357,.4347,24.2]),
            'TN':2000,
            'trunc_order':2,
            'z_forward':False,
            'i_forward':False,
            'i_bad_dx':[False,True,False,False],
            'max_iter':20,
            'rtol':1e-12,
            'atol':1e-12,
            'rel_tol':1e-9,
            'save_fig':False}

    
    system1 = rsp(idx=0,model_name='gw0',**kws1)
    system2 = rsp(idx=1,model_name='gw1',**kws1)

    a11 = StrongCoupling2(system1,system2,
                         #recompute_list=['p_data_tg0','p_data_tg1',
                         # 'h_data_tg0','h_data_tg1'],
                         #recompute_list=recompute_list,
                         NH=201)



if __name__ == "__main__":
    
    main()
