"""
Example: Thalamic model from Wilson and Ermentrout RSTA 2019,
Rubin and Terman JCNS 2004

"""

import numpy as np
import sympy as sym
#import matplotlib.pyplot as plt

from sympy import Matrix

from StrongCoupling import StrongCoupling

def rhs(t,z,pdict,option='val'):
        """
        Right-hand side of the Thalamic model from Wilson and Ermentrout
        RSTA 2019 and Rubin and Terman JCNS 2004
        
        
        Parameters
        
            t : float or sympy object.
                time
            z : array or list of floats or sympy objects.
                state variables of the thalamic model v, h, r, w.
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
        
        if option == 'val':
            exp = np.exp
        else:
            exp = sym.exp
        
        v, h, r, w = z
        
        v *= 100
        r /= 100
        
        ah = 0.128*exp(-(v+46)/18)  #
        bh = 4/(1+exp(-(v+23)/5))  #
        
        minf = 1/(1+exp(-(v+37)/7))  #
        hinf = 1/(1+exp((v+41)/4))  #
        rinf = 1/(1+exp((v+84)/4))  #
        pinf = 1/(1+exp(-(v+60)/6.2))  #
        #print(pinf)
        
        tauh = 1/(ah+bh)  #
        taur = 28+exp(-(v+25)/10.5)  #
        
        iL = pdict['gL']*(v-pdict['eL'])  #
        ina = pdict['gna']*(minf**3)*h*(v-pdict['ena'])  #
        ik = pdict['gk']*((0.75*(1-h))**4)*(v-pdict['ek'])  #
        it = pdict['gt']*(pinf**2)*r*(v-pdict['et'])  #
        
        
        dv = (-iL-ina-ik-it+pdict['ib'])/pdict['c']
        dh = (hinf-h)/tauh
        dr = (rinf-r)/taur
        dw = pdict['alpha']*(1-w)/(1+exp(-(v-pdict['vt'])/pdict['sigmat']))\
            -pdict['beta']*w
        #dw = alpha*(1-w)-beta*w
        
        if option == 'val':
            return np.array([dv/100,dh,dr*100,dw])
            #return np.array([dv,dh,dr])
        else:
            return Matrix([dv/100,dh,dr*100,dw])
            #return Matrix([dv,dh,dr])

def coupling(vars_pair,pdict,option='val'):
        """
        
        Synaptic coupling function between Thalamic oscillators.
        
        E.g.,this Python function is the function $G(x_i,x_j)$
        in the equation
        $\\frac{dx_i}{dt} = F(x_i) + \\varepsilon G(x_i,x_j)$
        
        Parameters
        
            vars_pair : list or array
                contains state variables from oscillator A and B, e.g.,
                vA, hA, rA, wA, vB, hB, rB, wB  
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
                returns numpy array if option == 'val'. 
                returns sympy Matrix if option == 'sym'
    
        """
        vA, hA, rA, wA, vB, hB, rB, wB = vars_pair

        if option == 'val':
            return -np.array([wB*(vA-pdict['esyn']),0,0,0])/pdict['c']
        else:
            return -Matrix([wB*(vA-pdict['esyn']),0,0,0])/pdict['c']

def main():
    
    var_names = ['v','h','r','w']
    
    pardict = {'gL_val':0.05,
               'gna_val':3,
               'gk_val':5,
               'gt_val':5,
               'eL_val':-70,
               'ena_val':50,
               'ek_val':-90,
               'et_val':0,
               'esyn_val':0,
               'c_val':1,
               'alpha_val':3,
               'beta_val':2,
               'sigmat_val':0.8,
               'vt_val':-20,
               'ib_val':3.5}
    
    kwargs = {'recompute_LC':True,
              'recompute_monodromy':True,
              'recompute_g_sym':True,
              'recompute_g':True,
              'recompute_het_sym':True,
              'recompute_z':True,
              'recompute_i':True,
              'recompute_k_sym':True,
              'recompute_p_sym':True,
              'recompute_p':True,
              'recompute_h_sym':True,
              'recompute_h':True,
              'i_bad_dx':True,
              'trunc_order':2,
              'dir':'home+thalamic_dat/',
              'NA':1001,
              'NB':1001,
              'p_iter':25,
              'TN':20000,
              'rtol':1e-7,
              'atol':1e-7,
              'rel_tol':1e-6,
              'method':'LSODA',
              'load_all':True}
    
    T_init = 10.6
    LC_init = np.array([-.64,0.71,0.25,0,T_init])
    
    a = StrongCoupling(rhs,coupling,LC_init,var_names,pardict,**kwargs)
    
    
if __name__ == "__main__":
    
    __spec__ = None

    main()
