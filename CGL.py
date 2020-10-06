# file for comparing to CGL. implement adjoint methods in Wilson 2020

# https://stackoverflow.com/questions/49306092/parsing-a-symbolic-expression-that-includes-user-defined-functions-in-sympy

# TODO: fix solutions scaled in amplitdue by 2pi. (5/27/2020)

"""
The logical flow of the class follows the paper by Wilson 2020.
-produce heterogeneous terms for g for arbirary dx
-substitute dx with g=g0 + psi*g1 + psi^2*g2+...
-produce het. terms for irc
-

this file is also practice for creating a more general class for any RHS.

TODO:
    -make sure that np.dot and sym matrix products are consistent.
    -check that np.identity and sym.eye are consistent

"""

from StrongCoupling import StrongCoupling

# user-defined

#import matplotlib
import numpy as np
from sympy import Matrix

def rhs(t,z,pdict,option='value'):
    """
    right-hand side of the equation of interest. CCGL model.
    
    write in standard python notation as if it will be used in an ODE solver.

    Returns
    -------
    right-hand side equauation in terms of the inputs. if x,y scalars,
    return scalar. If x,y, sympy symbols, return symbol.
    """
    
    x,y = z
    R2 = x**2 + y**2
    
    if option == 'value':
        return np.array([x*(1-R2)-pdict['q']*R2*y,
                         y*(1-R2)+pdict['q']*R2*x])
    elif option == 'sym':
        return Matrix([x*(1-R2)-pdict['q']*R2*y,
                       y*(1-R2)+pdict['q']*R2*x])

def coupling(vars_pair,pdict,option='value'):
        """
        r^(2n) to r^n function. default parameter order is from perspective of
        first oscillator.
        
        in this case the input is (x1,y1,x2,y2) and the output is an R^2 vec.
        """
        x1,y1,x2,y2 = vars_pair
        
        if option == 'value':
            return np.array([x2-x1-pdict['d']*(y2-y1),
                             y2-y1+pdict['d']*(x2-x1)])
        elif option == 'sym':
            return Matrix([x2-x1-pdict['d']*(y2-y1),
                           y2-y1+pdict['d']*(x2-x1)])
    

def main():
    
    var_names = ['x','y']
    
    pardict = {'q_val':1,
               'eps_val':0,
               'd_val':1}
    
    kwargs = {'recompute_LC':False,
              'recompute_monodromy':False,
              'recompute_g_sym':False,
              'recompute_g':False,
              'recompute_het_sym':False,
              'recompute_z':False,
              'recompute_i':False,
              'recompute_k_sym':False,
              'recompute_p_sym':False,
              'recompute_p':False,
              'recompute_h_sym':False,
              'recompute_h':False,
              'g_forward':False,
              'dir':'home+cgl_dat/',
              'trunc_order':9,
              'NA':501,
              'NB':501,
              'p_iter':25,
              'TN':20000,
              'rtol':1e-13,
              'atol':1e-13,
              'rel_tol':1e-10,
              'method':'LSODA',
              'load_all':True}
    
    T_init = 2*np.pi
    LC_init = np.array([1,0,T_init])
    
    # for NIC, 3rd derivatives go away, so we only need trunc_gh=3.
    a = StrongCoupling(rhs,coupling,LC_init,var_names,pardict,**kwargs)
    
    
    
if __name__ == "__main__":
    __spec__ = None
    main()
