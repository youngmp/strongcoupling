# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:43:11 2020

library for creating symbolic functions
"""
import MatchingLib as lib

import os
import math
import dill

import sympy as sym

from sympy import diff, Sum, Indexed, collect, expand
from sympy.utilities.lambdify import lambdify

def generate_expansions(obj):
    """
    generate expansions from Wilson 2020
    
    """
    print('* Generating expansions...')
    i_sym = sym.symbols('i_sym')  # summation index
    psi = obj.psi
    
    #obj.g_expand = {}
    for key in obj.var_names:
        sg = Sum(psi**i_sym*Indexed('g'+key,i_sym),(i_sym,1,obj.miter))
        sz = Sum(psi**i_sym*Indexed('z'+key,i_sym),(i_sym,0,obj.miter))
        si = Sum(psi**i_sym*Indexed('i'+key,i_sym),(i_sym,0,obj.miter))
        
        obj.g['expand_'+key] = sg.doit()
        obj.z['expand_'+key] = sz.doit()
        obj.i['expand_'+key] = si.doit()
    
    obj.z['vec'] = sym.zeros(obj.dim,1)
    
    for i,key in enumerate(obj.var_names):
        obj.z['vec'][i] = [obj.z['expand_'+key]]
    
    # rule to replace dv with gv, dh with gh, etc.
    obj.rule_d2g = {obj.dx_vec[i]:
                    obj.g['expand_'+k] for i,k in enumerate(obj.var_names)}
    
        
def load_coupling_expansions(obj,fn='gA',recompute=False):
    
    i = sym.symbols('i_sym')  # summation index
    #psi = obj.psi
    eps = obj.eps
    
    # for solution of isostables in terms of theta.
    obj.pA['expand'] = Sum(eps**i*Indexed('pA',i),(i,1,obj.miter)).doit()
    obj.pB['expand'] = Sum(eps**i*Indexed('pB',i),(i,1,obj.miter)).doit()
    
    
    #fname = obj.hodd['dat_fnames'][i]
    #file_does_not_exist = not(os.path.exists(fname))
    for key in obj.var_names:
        obj.g[key+'_epsA'] = []
        obj.g[key+'_epsB'] = []
        
        obj.i[key+'_epsA'] = []
        obj.i[key+'_epsB'] = []
        
    # check that files exist
        val = 0
        for key in obj.var_names:
            val += not(os.path.isfile(obj.g[key+'_epsA_fname']))
            val += not(os.path.isfile(obj.g[key+'_epsB_fname']))
            val += not(os.path.isfile(obj.i[key+'_epsA_fname']))
            val += not(os.path.isfile(obj.i[key+'_epsB_fname']))
            
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False
    
    if recompute or files_do_not_exist:
        generate_coupling_expansions(obj)
        
        for key in obj.var_names:
            # dump
            dill.dump(obj.g[key+'_epsA'],open(obj.g[key+'_epsA_fname'],'wb'),
                      recurse=True)
            dill.dump(obj.g[key+'_epsB'],open(obj.g[key+'_epsB_fname'],'wb'),
                      recurse=True)
            
            dill.dump(obj.i[key+'_epsA'],open(obj.i[key+'_epsA_fname'],'wb'),
                      recurse=True)
            dill.dump(obj.i[key+'_epsB'],open(obj.i[key+'_epsB_fname'],'wb'),
                      recurse=True)
        
    else:
        
        for key in obj.var_names:
            obj.g[key+'_epsA'] = dill.load(open(obj.g[key+'_epsA_fname'],'rb'))
            obj.g[key+'_epsB'] = dill.load(open(obj.g[key+'_epsB_fname'],'rb'))
            
            obj.i[key+'_epsA'] = dill.load(open(obj.i[key+'_epsA_fname'],'rb'))
            obj.i[key+'_epsB'] = dill.load(open(obj.i[key+'_epsB_fname'],'rb'))
        
    # vector of i expansion
    obj.i['vecA'] = sym.zeros(obj.dim,1)
    obj.i['vecB'] = sym.zeros(obj.dim,1)
    
    for i,key in enumerate(obj.var_names):
        #print('ia',i,key,obj.i[key+'_epsA'])
        #print('ib',i,key,obj.i[key+'_epsB'])
        obj.i['vecA'][i] = obj.i[key+'_epsA']
        obj.i['vecB'][i] = obj.i[key+'_epsB']
    
    pass

def generate_coupling_expansions(obj,verbose=True):
    """
    generate expansions for coupling.
    """
    if verbose:
        print('* Generating coupling expansions...')
    i = sym.symbols('i_sym')  # summation index
    psi = obj.psi
    eps = obj.eps
    
    
    ruleA = {'psi':obj.pA['expand']}
    ruleB = {'psi':obj.pB['expand']}
    
    rule_trunc = {}
    for k in range(obj.miter,obj.miter+200):
        rule_trunc.update({obj.eps**k:0})
    
    for key in obj.var_names:
        if verbose:
            print('key in coupling expand',key)
        gA = Sum(psi**i*Indexed('g'+key+'A',i),(i,1,obj.miter)).doit()
        gB = Sum(psi**i*Indexed('g'+key+'B',i),(i,1,obj.miter)).doit()
        
        iA = Sum(psi**i*Indexed('i'+key+'A',i),(i,0,obj.miter)).doit()
        iB = Sum(psi**i*Indexed('i'+key+'B',i),(i,0,obj.miter)).doit()
        
        tmp = gA.subs(ruleA)
        tmp = sym.expand(tmp,basic=False,deep=True,
                         power_base=False,power_exp=False,
                         mul=False,log=False,
                         multinomial=True)
        tmp = tmp.subs(rule_trunc)
        gA_collected = collect(expand(tmp).subs(rule_trunc),eps)
        if verbose:
            print('completed gA collected')
        
        tmp = gB.subs(ruleB)
        tmp = sym.expand(tmp,basic=False,deep=True,
                         power_base=False,power_exp=False,
                         mul=False,log=False,
                         multinomial=True)
        tmp = tmp.subs(rule_trunc)
        gB_collected = collect(expand(tmp).subs(rule_trunc),eps)
        if verbose:
            print('completed gB collected')
        
        tmp = iA.subs(ruleA)
        tmp = sym.expand(tmp,basic=False,deep=True,
                         power_base=False,power_exp=False,
                         mul=False,log=False,
                         multinomial=True)
        tmp = tmp.subs(rule_trunc)
        iA_collected = collect(expand(tmp).subs(rule_trunc),eps)
        if verbose:
            print('completed iA collected')
            
        tmp = iB.subs(ruleB)
        tmp = sym.expand(tmp,basic=False,deep=True,
                         power_base=False,power_exp=False,
                         mul=False,log=False,
                         multinomial=True)
        tmp = tmp.subs(rule_trunc)
        iB_collected = collect(expand(tmp).subs(rule_trunc),eps)
        if verbose:
            print('completed iB collected')
        
        
        obj.g[key+'_epsA'] = 0
        obj.g[key+'_epsB'] = 0

        obj.i[key+'_epsA'] = 0
        obj.i[key+'_epsB'] = 0
        
        for j in range(obj.miter):
            obj.g[key+'_epsA'] += eps**j*gA_collected.coeff(eps,j)
            obj.g[key+'_epsB'] += eps**j*gB_collected.coeff(eps,j)
            
            obj.i[key+'_epsA'] += eps**j*iA_collected.coeff(eps,j)
            obj.i[key+'_epsB'] += eps**j*iB_collected.coeff(eps,j)
            

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
        print('g sym deriv order=',i)
        p = lib.kProd(i,obj.dx_vec)
        for j,key in enumerate(obj.var_names):
            print('\t var=',key)
            d = lib.vec(lib.df(obj.rhs_sym[j],obj.x_vec,i))
            
            het[key] += (1/math.factorial(i))*p.dot(d)
            #print((1/math.factorial(i)))
            
            #print(het[key])
            
            
            
    #print(h)
    out = {}
    
    #  collect in psi.
    rule = {**obj.rule_g0,**obj.rule_d2g}
    
    rule_trunc = {}
    for k in range(obj.miter,obj.miter+200):
        rule_trunc.update({obj.psi**k:0})
            
    for key in obj.var_names:
        
        print('g sym subs key=',key)
        # remove small floating points
        tmp = het[key].subs(rule)
        
        #print(tmp)
        #print(rule)
        # collect expressions
        print('g sym expand 1 key=',key)
        tmp = sym.expand(tmp,basic=False,deep=True,
                         power_base=False,power_exp=False,
                         mul=False,log=False,
                         multinomial=True)
        
        tmp = tmp.subs(rule_trunc)
        
        print('g sym collect 1 key=',key)
        tmp = sym.collect(tmp,obj.psi).subs(rule_trunc)
        
        print('g sym expand 2 key=',key)
        tmp = sym.expand(tmp).subs(rule_trunc)
        
        #print(tmp)
        print('g sym collect 2 key=',key)
        tmp = sym.collect(tmp,obj.psi).subs(rule_trunc)
        
        out[key] = tmp
        #print()
        #print(tmp)
        
    return out

def load_jac_sym(obj):
    # symbol J on LC.
    obj.jac_sym = sym.zeros(obj.dim,obj.dim)
    
    for i in range(obj.dim):
        for j in range(obj.dim):
            fn = obj.rhs_sym[i]
            var = obj.var_names[j]
            
            obj.jac_sym[i,j] = diff(fn,var)
    
    rule = {**obj.rule_LC,**obj.rule_par}
    # callable jacobian matrix evaluated along limit cycle
    obj.jacLC = lambdify((obj.t),obj.jac_sym.subs(rule))