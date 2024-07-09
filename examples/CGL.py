
# https://stackoverflow.com/questions/49306092/parsing-a-symbolic-expression
# -that-includes-user-defined-functions-in-sympy

"""
Example: Complex Ginzburgh-Landau (CGL) model from Wilson and Ermentrout RSTA
2019

"""

from StrongCoupling import StrongCoupling

# user-defined

#import matplotlib
import numpy as np
from sympy import Matrix

def rhs(t, z, pdict, option='value'):
    """
    Right-hand side of the Complex Ginzburgh-Landau (CGL) model from
    Wilson and Ermentrout RSTA 2019 

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

    x,y = z
    R2 = x**2 + y**2
    q = pdict['q']

    if option == 'value':
        return np.array([x*(1-R2)-q*R2*y,y*(1-R2)+q*R2*x])
    elif option == 'sym':
        return Matrix([x*(1-R2)-q*R2*y,y*(1-R2)+q*R2*x])

def coupling(vars_pair, pdict, option='value'):
    """

    Diffusive coupling function between Complex Ginzburgh Landau
    (CGL) oscillators.

    E.g.,this Python function is the function $G(x_i,x_j)$
    in the equation
    $\\frac{dx_i}{dt} = F(x_i) + \\varepsilon G(x_i,x_j)$

    Parameters

        vars_pair : list or array
            contains state variables from oscillator A and B, e.g.,
            x1,y1,x2,y2
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
              'recompute_monodromy':True,
              'recompute_g_sym':False,
              'recompute_g':False,
              'recompute_het_sym':False,
              'recompute_z':False,
              'recompute_i':False,
              'recompute_k_sym':False,
              'recompute_p_sym':False,
              'recompute_p':True,
              'recompute_h_sym':False,
              'recompute_h':True,
              'g_forward':False,
              'z_forward':False,
              'i_forward':[True,False,False,False,False,False,False,False,False,False],
              'dense':True,
              'dir':'home+cgl_dat_strongcoup/',
              'trunc_order':4,
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
