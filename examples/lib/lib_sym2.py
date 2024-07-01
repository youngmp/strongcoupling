# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:43:11 2020

library for creating symbolic functions
"""

from . import lib

import os
import math
import dill

import sympy as sym

from sympy import diff, Sum, Indexed, collect, expand, symbols
from sympy.utilities.lambdify import lambdify

import logging # for debugging

def generate_expansions(obj):
    """
    generate expansions from Wilson 2020
    """
    
    logging.info('* Generating expansions...')
    i_sym = sym.symbols('i_sym')  # summation index
    psi = obj.psi
    
    # give the raw variables a name for easier substitutions later
    # for example, {Indexed('g'+key,i_sym):some_implented_function}
    
    obj.g_og = []
    obj.z_og = []
    obj.i_og = []
    
    for key in obj.var_names:
        for i in range(obj.miter):
            obj.g_og.append(Indexed('g'+key,i))
            obj.z_og.append(Indexed('z'+key,i))
            obj.i_og.append(Indexed('i'+key,i))
    
    for key in obj.var_names:
        sg = Sum(psi**i_sym*Indexed('g'+key,i_sym),(i_sym,1,obj.miter-1))
        sz = Sum(psi**i_sym*Indexed('z'+key,i_sym),(i_sym,0,obj.miter-1))
        si = Sum(psi**i_sym*Indexed('i'+key,i_sym),(i_sym,0,obj.miter-1))
        
        
        
        obj.g['expand_'+key] = sg.doit()
        obj.z['expand_'+key] = sz.doit()
        obj.i['expand_'+key] = si.doit()
    
    obj.z['vec_psi'] = sym.zeros(obj.dim,1)
    obj.i['vec_psi'] = sym.zeros(obj.dim,1)
    
    for i,key in enumerate(obj.var_names):
        obj.z['vec_psi'][i] = [obj.z['expand_'+key]]
        obj.i['vec_psi'][i] = [obj.i['expand_'+key]]
    
    # for computing derivatives
    obj.dx_vec = sym.zeros(1,obj.dim) 
    obj.x_vec = sym.zeros(obj.dim,1)
    
    for i,name in enumerate(obj.var_names):
        # save var1, var2, ..., varN
        symname = symbols(name)
        obj.x_vec[i] = symname
        
        # save dvar1, dvar2, ..., dvarN
        symd = symbols('d'+name)
        obj.dx_vec[i] = symd
            
    # rule to replace dv with gv, dh with gh, etc.
    obj.rule_d2g = {obj.dx_vec[i]:
                    obj.g['expand_'+k] for i,k in enumerate(obj.var_names)}

def load_coupling_expansions(obj,recompute=False):
    """
    compute expansions related to the coupling function
    """
    k = sym.symbols('i_sym')  # summation index
    eps = obj.eps
    
    # for solution of isostables in terms of theta.
    obj.p['expand'] = Sum(eps**k*Indexed('p_'+obj.model_name,k),
                          (k,1,obj.miter)).doit()
    
    
    for key in obj.var_names:
        obj.g[key+'_eps'] = []
        obj.i[key+'_eps'] = []
        obj.z[key+'_eps'] = []
        
        # check that files exist
        val = 0
        for key in obj.var_names:
            val += not(os.path.isfile(obj.g[key+'_eps_fname']))
            val += not(os.path.isfile(obj.i[key+'_eps_fname']))
            val += not(os.path.isfile(obj.z[key+'_eps_fname']))
            
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False
    
    if recompute or files_do_not_exist:
        
        generate_coupling_expansions(obj)

        for key in obj.var_names:
            # dump
            v = obj.g[key+'_eps_fname']
            dill.dump(obj.g[key+'_eps'],
                      open(v,'wb'),recurse=True)
                
            v = obj.i[key+'_eps_fname']
            dill.dump(obj.i[key+'_eps'],
                      open(v,'wb'),recurse=True)
            
            v = obj.z[key+'_eps_fname']
            dill.dump(obj.z[key+'_eps'],
                      open(v,'wb'),recurse=True)
                
    else:

        for key in obj.var_names:
            v = obj.g[key+'_eps_fname']
            obj.g[key+'_eps'] = dill.load(open(v,'rb'))
            
            v = obj.i[key+'_eps_fname']
            obj.i[key+'_eps'] = dill.load(open(v,'rb'))
            
            v = obj.z[key+'_eps_fname']
            obj.z[key+'_eps'] = dill.load(open(v,'rb'))
            
    # vector of i expansion
    obj.i['vec'] = sym.zeros(obj.dim,1)
    obj.z['vec'] = sym.zeros(obj.dim,1)

    for key_idx,key in enumerate(obj.var_names):
        obj.i['vec'][key_idx] = obj.i[key+'_eps']
        obj.z['vec'][key_idx] = obj.z[key+'_eps']

def generate_coupling_expansions(obj):
    """
    obj: nBodyCoupling object
    generate expansions for coupling.
    """
    
    logging.info('* Generating coupling expansions...')
    k = sym.symbols('i_sym')  # summation index
    psi = obj.psi;eps = obj.eps

    rule_psi = {'psi':obj.p['expand']}
    
    rule_trunc = {}
    for l in range(obj.miter,obj.miter+200):
        rule_trunc.update({obj.eps**l:0})
    
    for key in obj.var_names:
        logging.info('key in coupling expand '+key)

        g_sum = Sum(psi**k*Indexed('g_'+obj.model_name+'_'+key,k),
                    (k,1,obj.miter)).doit()
        z_sum = Sum(psi**k*Indexed('z_'+obj.model_name+'_'+key,k),
                    (k,0,obj.miter)).doit()
        i_sum = Sum(psi**k*Indexed('i_'+obj.model_name+'_'+key,k),
                    (k,0,obj.miter)).doit()
        
        tmp = g_sum.subs(rule_psi)
        tmp = sym.expand(tmp,basic=True,deep=True,
                         power_base=False,power_exp=False,
                         mul=True,log=False,
                         multinomial=True)
        tmp = tmp.subs(rule_trunc)
        
        g_collected = collect(expand(tmp).subs(rule_trunc),eps)        

        tmp = z_sum.subs(rule_psi)
        tmp = sym.expand(tmp,basic=True,deep=True,
                         power_base=False,power_exp=False,
                         mul=True,log=False,
                         multinomial=True)
        tmp = tmp.subs(rule_trunc)
        z_collected = collect(expand(tmp).subs(rule_trunc),eps)
        
        tmp = i_sum.subs(rule_psi)
        tmp = sym.expand(tmp,basic=True,deep=True,
                         power_base=False,power_exp=False,
                         mul=True,log=False,
                         multinomial=True)
        tmp = tmp.subs(rule_trunc)
        i_collected = collect(expand(tmp).subs(rule_trunc),eps)


        obj.g[key+'_eps'] = 0
        obj.z[key+'_eps'] = 0
        obj.i[key+'_eps'] = 0
            
        for l in range(obj.miter):
            obj.g[key+'_eps'] += eps**l*g_collected.coeff(eps,l)
            obj.z[key+'_eps'] += eps**l*z_collected.coeff(eps,l)
            obj.i[key+'_eps'] += eps**l*i_collected.coeff(eps,l)

            
def generate_coupling_expansions_old(obj):
    """
    obj: nBodyCoupling object
    generate expansions for coupling.
    """
    
    logging.info('* Generating coupling expansions...')
    k = sym.symbols('i_sym')  # summation index
    psi = obj.psi
    eps = obj.eps

    rule_psi = {}
    for i in range(obj.N):
        rule_psi[i] = {'psi':obj.p[i]['expand']}
    
    rule_trunc = {}
    for l in range(obj.miter,obj.miter+200):
        rule_trunc.update({obj.eps**l:0})
    
    for key in obj.var_names:
        logging.info('key in coupling expand '+key)
            
        for i in range(obj.N):

            g_sum = Sum(psi**k*Indexed('g'+key+str(i),k),(k,1,obj.miter)).doit()
            z_sum = Sum(psi**k*Indexed('z'+key+str(i),k),(k,0,obj.miter)).doit()
            i_sum = Sum(psi**k*Indexed('i'+key+str(i),k),(k,0,obj.miter)).doit()
            
        
            tmp = g_sum.subs(rule_psi[i])
            tmp = sym.expand(tmp,basic=True,deep=True,
                             power_base=False,power_exp=False,
                             mul=True,log=False,
                             multinomial=True)
            tmp = tmp.subs(rule_trunc)
            
            g_collected = collect(expand(tmp).subs(rule_trunc),eps)
        
        
            tmp = z_sum.subs(rule_psi[i])
            tmp = sym.expand(tmp,basic=True,deep=True,
                             power_base=False,power_exp=False,
                             mul=True,log=False,
                             multinomial=True)
            tmp = tmp.subs(rule_trunc)
            z_collected = collect(expand(tmp).subs(rule_trunc),eps)

            tmp = i_sum.subs(rule_psi[i])
            tmp = sym.expand(tmp,basic=True,deep=True,
                             power_base=False,power_exp=False,
                             mul=True,log=False,
                             multinomial=True)
            tmp = tmp.subs(rule_trunc)
            i_collected = collect(expand(tmp).subs(rule_trunc),eps)
                    
            obj.g[i][key+'_eps'] = 0
            obj.z[i][key+'_eps'] = 0
            obj.i[i][key+'_eps'] = 0
            
            for l in range(obj.miter):
                obj.g[i][key+'_eps'] += eps**l*g_collected.coeff(eps,l)
                obj.z[i][key+'_eps'] += eps**l*z_collected.coeff(eps,l)
                obj.i[i][key+'_eps'] += eps**l*i_collected.coeff(eps,l)


    return obj.g,obj.i


def generate_g_sym(obj):
    """
    generate heterogeneous terms for the Floquet eigenfunctions g.
    
    purely symbolic.

    Returns
    -------
    list of symbolic heterogeneous terms in self.ghx_sym, self.ghy_sym.

    """
    # get the general expression for h before plugging in g.
    het = {key: sym.sympify(0) for key in obj.var_names}
    
    for i in range(2,obj.miter):
        logging.info('g sym deriv order='+str(i))
        p = lib.kProd(i,obj.dx_vec)
        for j,key in enumerate(obj.var_names):
            logging.info('\t var='+str(key))
            d = lib.vec(lib.df(obj.rhs_sym[j],obj.x_vec,i))
            
            het[key] += (1/math.factorial(i))*p.dot(d)
            
    out = {}
    
    #  collect in psi.
    rule = {**obj.rule_g0,**obj.rule_d2g}
    
    rule_trunc = {}
    for k in range(obj.miter,obj.miter+200):
        rule_trunc.update({obj.psi**k:0})
            
    for key in obj.var_names:
        
        logging.info('g sym subs key='+str(key))
        # remove small floating points
        tmp = het[key].subs(rule)
        
        # collect expressions
        logging.info('g sym expand 1 key='+str(key))
        
        tmp = sym.expand(tmp,basic=False,deep=True,
                         power_base=False,power_exp=False,
                         mul=False,log=False,
                         multinomial=True)
        
        tmp = tmp.subs(rule_trunc)
        
        logging.info('g sym collect 1 key='+str(key))
        tmp = sym.collect(tmp,obj.psi).subs(rule_trunc)
        
        logging.info('g sym expand 2 key='+str(key))
        tmp = sym.expand(tmp).subs(rule_trunc)
        
        logging.info('g sym collect 2 key='+str(key))
        tmp = sym.collect(tmp,obj.psi).subs(rule_trunc)
        
        out[key] = tmp
        
    return out

def load_jac_sym(obj):
    """
    obj: nBodyCoupling object
    lambdify the jacobian matrix evaluated along limit cycle
    """
    # symbol J on LC.
    obj.jac_sym = sym.zeros(obj.dim,obj.dim)
    
    for i in range(obj.dim):
        for j in range(obj.dim):
            fn = obj.rhs_sym[i]
            #var = obj.var_names[j]
            var = obj.syms[j]

            tmp = sym.powsimp(diff(fn,var))
            
            obj.jac_sym[i,j] = tmp
    
    rule = {**obj.rule_lc_local,**obj.rule_par}
    # callable jacobian matrix evaluated along limit cycle
    
    obj.jaclc = lambdify((obj.t),obj.jac_sym.subs(rule))
