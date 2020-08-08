# file for comparing to CGL. implement adjoint methods in Wilson 2020

# https://stackoverflow.com/questions/49306092/...
#parsing-a-symbolic-expression-that-includes-user-defined-functions-in-sympy

# TODO: note heterogeneous terms in right-hand sides are hard-coded.

"""
The logical flow of the class follows the paper by Wilson 2020.
-produce heterogeneous terms for g for arbirary dx
-substitute dx with g=g0 + psi*g1 + psi^2*g2+...
-produce het. terms for irc
-...

this file is also practice for creating a more general class for any RHS.

coupling functions for thalamic neurons from RTSA Ermentrout, Park, Wilson 2019

"""

# user-defined
import MatchingLib as lib
from interp_basic import interp_basic as interpb
from interp2d_basic import interp2d_basic as interp2db

import inspect
import time
import os
import math
#import time
import dill
import copy
import matplotlib

import scipy.interpolate as si
import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sympy import Matrix, symbols,diff, pi, Sum, Indexed, collect, expand
#from sympy import Function
from sympy import sympify as s
from sympy.physics.quantum import TensorProduct as kp
from sympy.utilities.lambdify import lambdify, implemented_function
#from sympy.interactive import init_printing

imp_fn = implemented_function

#from interpolate import interp1d
#from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import interp2d, Rbf
from scipy.integrate import solve_ivp, quad

matplotlib.rcParams.update({'figure.max_open_warning': 0})


class Thalamic(object):
    """
    Thalamic model from RSTA 2019
    Requires sympy, numpy, matplotlib.
    """
    
    def __init__(self,**kwargs):

        """
        recompute_g_sym : recompute heterogenous terms for Floquet e.funs g
        
        """

        defaults = {
            'gL_val':0.05,
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

            'trunc_order':3,
            'trunc_derivative':2,
            
            'TN':20000,
            'dir':'dat',
            
            'recompute_LC':False,
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
            'load_all':True,
            }
        
        
        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # variable names
        self.var_names = ['v','h','r','w']
        self.n = len(self.var_names)
        
        # misc variables
        self.miter = self.trunc_order+1

        # Symbolic variables and functions
        self.eye = np.identity(self.n)
        self.psi, self.eps, self.kap_sym = sym.symbols('psi eps kap_sym')
        
        self.rtol = 1e-9
        self.atol = 1e-9
        self.method = 'LSODA'
        self.rel_tol = 1e-9
        
        # for coupling computation. ctrl+F to see usage.
        self.NA = np.array([10,100,200,400,800,1000])[:self.trunc_order+1]
        self.NB = self.NA + 1
        self.Ns = np.array([10,1000,2000,3000,6000,10000])[:self.trunc_order+1]
        self.smax = np.array([10,300,400,500,600,1000])[:self.trunc_order+1]
        
        
        #self.x1,self.x2,self.x3,self.t = symbols('x1 x2,x3, t')
        #self.x,self.y,self.z,self.t = symbols('x1 x2,x3, t')
        #self.f_list, self.x_vars, self.y_vars, self.z_vars = ([] for i in range(4))
        
        # parameters
        self.c = symbols('c')
        self.alpha, self.beta = symbols('alpha beta')
        self.vt, self.sigmat = symbols('vt sigmat')
        self.gL, self.eL = symbols('gL eL')
        self.gna, self.ena = symbols('gna, ena')
        self.gk, self.ek = symbols('gk ek')
        self.gt, self.et = symbols('gt et')
        self.esyn = symbols('esyn')
        
        
        self.rule_par = {self.c:self.c_val,
                         self.alpha:self.alpha_val,
                         self.beta:self.beta_val,
                         self.vt:self.vt_val,
                         self.sigmat:self.sigmat_val,
                         self.gL:self.gL_val,
                         self.eL:self.eL_val,
                         self.gna:self.gna_val,
                         self.ena:self.ena_val,
                         self.gk:self.gk_val,
                         self.ek:self.ek_val,
                         self.esyn:self.esyn_val,
                         self.gt:self.gt_val,
                         self.et:self.et_val}
        

        # single-oscillator variables
        
        self.v, self.h, self.r, self.w, self.t, self.s = symbols('v h r w t s')
        self.tA, self.tB, = symbols('tA tB')
        self.dv, self.dh, self.dr, self.dw = symbols('dv dh dr dw')
        
        # coupling variables
        self.thA, self.psiA, self.thB, self.psiB = symbols('thA psiA thB psiB')
        
        self.vA, self.hA, self.rA, self.wA = symbols('vA hA rA wA')
        self.vB, self.hB, self.rB, self.wB = symbols('vB hB rB wB')

        self.dvA, self.dhA, self.drA, self.dwA = symbols('dvA dhA drA dwA')
        self.dvB, self.dhB, self.drB, self.dwB = symbols('dvB dhB drB dwB')

        self.vars = [self.v,self.h,self.r,self.w]
        self.A_vars = [self.vA,self.hA,self.rA,self.wA]
        self.dA_vars = [self.dvA,self.dhA,self.drA,self.dwA]
        
        self.B_vars = [self.vB,self.hB,self.rB,self.wB]
        self.dB_vars = [self.dvB,self.dhB,self.drB,self.dwB]
        
        self.A_pair = Matrix([[self.vA,self.hA,self.rA,self.wA,
                               self.vB,self.hB,self.rB,self.wB]])
        
        self.dA_pair = Matrix([[self.dvA,self.dhA,self.drA,self.dwA,
                                self.dvB,self.dhB,self.drB,self.dwB]])
        
        self.B_pair = Matrix([[self.vB,self.hB,self.rB,self.wB,
                               self.vA,self.hA,self.rA,self.wA]])
        
        self.dB_pair = Matrix([[self.dvB,self.dhB,self.drB,self.dwB,
                                self.dvA,self.dhA,self.drA,self.dwA]])
        
        self.d = {'v':self.dv,'h':self.dh,'r':self.dr,'w':self.dw}
        
        self.dx_vec = Matrix([[self.dv,self.dh,self.dr,self.dw]])
        self.x_vec = Matrix([[self.v],[self.h],[self.r],[self.w]])
        
        
        # function dicts
        # individual functions
        self.LC = {}
        self.g = {}
        self.het1 = {}
        self.z = {}
        self.i = {}
        
        # for coupling
        self.cA = {}
        self.cB = {}
        
        self.kA = {}
        self.kB = {}
        
        self.pA = {}
        self.pB = {}
        self.hodd = {}
        self.het2 = {}
        
        
        # filenames and directories
        self.dir = 'thalamic_dat/'
        
        if (not os.path.exists(self.dir)):
            os.makedirs(self.dir)
        
        lib.generate_fnames(self)
        #self.generate_fnames()
        
        # make rhs callable
        
        self.thal_sym = self.thal_rhs(0,[self.v,self.h,self.r,self.w],
                                      option='sym')
        

        # symbol J on LC.
        self.jac_sym = sym.zeros(self.n,self.n)
        
        for i in range(self.n):
            for j in range(self.n):
                fn = self.thal_sym[i]
                var = self.var_names[j]
                
                self.jac_sym[i,j] = diff(fn,var)
        
        
        #print('jac sym',self.jac_sym[0,0])
      
        # assume gx is the first coordinate of Floquet eigenfunction g
        # brackets denote Taylor expansion functions
        # now substitute Taylor expansion dx = gx[0] + gx[1] + gx[2] + ...
        
        self.generate_coupling_expansions()
        self.generate_expansions()
        
        self.load_limit_cycle()
            
        self.rule_LC = {}
        for i,key in enumerate(self.var_names):
            self.rule_LC.update({self.vars[i]:self.LC['imp_'+key](self.t)})

        #self.rule_LC = {self.v:self.LC['imp_v'](self.t),
        #                self.h:self.LC['imp_h'](self.t),
        #                self.r:self.LC['imp_r'](self.t),
        #                self.w:self.LC['imp_w'](self.t)}
            
        rule = {**self.rule_LC,**self.rule_par}
            
        # callable jacobian matrix evaluated along limit cycle
        self.jacLC = lambdify((self.t),self.jac_sym.subs(rule))
        
        
        if self.load_all:
            
            
            # get monodromy matrix
            self.load_monodromy()
            
            # get heterogeneous terms for g, floquet e. fun.
            self.load_g_sym()
                
            # get g
            self.load_g()
            
            # get het. terms for z and i
            self.load_het_sym()
            
            # get iPRC, iIRC.
            self.load_z()
            self.load_i()
            
            
            self.load_k_sym()
            
            self.load_p_sym()
            self.load_p()
        
            self.load_h_sym()
            self.load_h()
        
            
            
        
    def thal_rhs(self,t,z,option='val'):
        """
        right-hand side of the equation of interest. thalamic neural model.
        """
        
        v, h, r, w = z
        
        v *= 100
        r /= 100
        
        if option == 'val':
            gL = self.gL_val
            eL = self.eL_val
            gna = self.gna_val
            ena = self.ena_val
            gk = self.gk_val
            ek = self.ek_val
            gt = self.gt_val
            et = self.et_val
            c = self.c_val
            alpha = self.alpha_val
            vt = self.vt_val
            sigmat = self.sigmat_val
            beta = self.beta_val
            exp = np.exp

        elif option == 'sym':
            gL = self.gL
            eL = self.eL
            gna = self.gna
            ena = self.ena
            gk = self.gk
            ek = self.ek
            gt = self.gt
            et = self.et
            c = self.c
            alpha = self.alpha
            vt = self.vt
            sigmat = self.sigmat
            beta = self.beta
            exp = sym.exp
            
        else:
            raise ValueError('Unrecognized option',option)
            
        """
        ah = 0.128*exp(-((v*100)+46)/18)  #
        bh = 4/(1+exp(-((v*100)+23)/5))  #
        
        minf = 1/(1+exp(-((v*100)+37)/7))  #
        hinf = 1/(1+exp(((v*100)+41)/4))  #
        rinf = 1/(1+exp(((v*100)+84)/4))  #
        pinf = 1/(1+exp(-((v*100)+60)/6.2))  #
        
        tauh = 1/(ah+bh)  #
        taur = 28+exp(-((v*100)+25)/10.5)  #
        
        iL = gL*((v*100)-eL)  #
        ina = gna*(minf**3)*h*((v*100)-ena)  #
        ik = gk*((0.75*(1-h))**4)*((v*100)-ek)  #
        it = gt*(pinf**2)*(r/100)*((v*100)-et)  #
        ib = 3.5
        
        dv = (-iL-ina-ik-it+ib)/c
        dh = (hinf-h)/tauh
        dr = (rinf-(r/100))/taur
        dw = alpha*(1-w)/(1+exp(-((v*100)-vt)/sigmat))-beta*w
        """
        
        ah = 0.128*exp(-(v+46)/18)  #
        bh = 4/(1+exp(-(v+23)/5))  #
        
        minf = 1/(1+exp(-(v+37)/7))  #
        hinf = 1/(1+exp((v+41)/4))  #
        rinf = 1/(1+exp((v+84)/4))  #
        pinf = 1/(1+exp(-(v+60)/6.2))  #
        #print(pinf)
        
        tauh = 1/(ah+bh)  #
        taur = 28+exp(-(v+25)/10.5)  #
        
        iL = gL*(v-eL)  #
        ina = gna*(minf**3)*h*(v-ena)  #
        ik = gk*((0.75*(1-h))**4)*(v-ek)  #
        it = gt*(pinf**2)*r*(v-et)  #
        ib = 3.5
        
        dv = (-iL-ina-ik-it+ib)/c
        dh = (hinf-h)/tauh
        dr = (rinf-r)/taur
        dw = alpha*(1-w)/(1+exp(-(v-vt)/sigmat))-beta*w
        #dw = alpha*(1-w)-beta*w
        
        if option == 'val':
            return np.array([dv/100,dh,dr*100,dw])
            #return np.array([dv,dh,dr])
        else:
            return Matrix([dv/100,dh,dr*100,dw])
            #return Matrix([dv,dh,dr])


    def numerical_jac(self,fn,x,eps=1e-7):
        """
        return numerical Jacobian function
        """
        n = len(x)
        J = np.zeros((n,n))
        
        PM = np.zeros_like(J)
        PP = np.zeros_like(J)
        
        for k in range(n):
            epsvec = np.zeros(n)
            epsvec[k] = eps
            PP[:,k] = fn(0,x+epsvec)
            PM[:,k] = fn(0,x-epsvec)
            
        J = (PP-PM)/(2*eps)
        
        return J
        
    def thal_coupling(self,vars_pair,option='val'):
        vA, hA, rA, wA, vB, hB, rB, wB = vars_pair

        if option == 'val':
            return -np.array([wB*(vA-self.esyn_val),0,0,0])/self.c_val
        else:
            return -Matrix([wB*(vA-self.esyn),0,0,0])/self.c

    def generate_expansions(self):
        """
        generate expansions from Wilson 2020
        """
        i_sym = sym.symbols('i_sym')  # summation index
        psi = self.psi
        
        #self.g_expand = {}
        for key in self.var_names:
            sg = Sum(psi**i_sym*Indexed('g'+key,i_sym),(i_sym,0,self.miter))
            sz = Sum(psi**i_sym*Indexed('z'+key,i_sym),(i_sym,0,self.miter))
            si = Sum(psi**i_sym*Indexed('i'+key,i_sym),(i_sym,0,self.miter))
            
            
            self.g['expand_'+key] = sg.doit()
            self.z['expand_'+key] = sz.doit()
            self.i['expand_'+key] = si.doit()
        
        self.z['vec'] = Matrix([[self.z['expand_v']],
                                [self.z['expand_h']],
                                [self.z['expand_r']],
                                [self.z['expand_w']]])
        
        # rule to replace dv with gv, dh with gh, etc.
        self.rule_d2g = {self.d[k]: self.g['expand_'+k] for k in self.var_names}
        
        #print('self.rule_d2g)',self.rule_d2g)
        #print('rule_d2g',self.rule_d2g)
    
    def load_coupling_expansions(self):
        # finish this function for speedup at initialization.
        pass
        
    def generate_coupling_expansions(self):
        """
        generate expansions for coupling.
        """
        
        i = sym.symbols('i_sym')  # summation index
        psi = self.psi
        eps = self.eps
        
        # for solution of isostables in terms of theta.
        self.pA['expand'] = Sum(eps**i*Indexed('pA',i),(i,1,self.miter)).doit()
        self.pB['expand'] = Sum(eps**i*Indexed('pB',i),(i,1,self.miter)).doit()
        
        ruleA = {'psi':self.pA['expand']}
        ruleB = {'psi':self.pB['expand']}
        
        for key in self.var_names:
            gA = Sum(psi**i*Indexed('g'+key+'A',i),(i,1,self.miter)).doit()
            gB = Sum(psi**i*Indexed('g'+key+'B',i),(i,1,self.miter)).doit()
            
            iA = Sum(psi**i*Indexed('i'+key+'A',i),(i,0,self.miter)).doit()
            iB = Sum(psi**i*Indexed('i'+key+'B',i),(i,0,self.miter)).doit()
            
            gA_collected = collect(expand(gA.subs(ruleA)),eps)
            gA_collected = collect(expand(gA_collected),eps)
            
            gB_collected = collect(expand(gB.subs(ruleB)),eps)
            gB_collected = collect(expand(gB_collected),eps)
            
            iA_collected = collect(expand(iA.subs(ruleA)),eps)
            iA_collected = collect(expand(iA_collected),eps)
            
            iB_collected = collect(expand(iB.subs(ruleB)),eps)
            iB_collected = collect(expand(iB_collected),eps)
            
            self.g[key+'_epsA'] = 0
            self.g[key+'_epsB'] = 0

            self.i[key+'_epsA'] = 0
            self.i[key+'_epsB'] = 0
            
            for j in range(self.miter):
                self.g[key+'_epsA'] += eps**j*gA_collected.coeff(eps,j)
                self.g[key+'_epsB'] += eps**j*gB_collected.coeff(eps,j)
                
                self.i[key+'_epsA'] += eps**j*iA_collected.coeff(eps,j)
                self.i[key+'_epsB'] += eps**j*iB_collected.coeff(eps,j)
                
    
        # vector of i expanstion
        self.i['vec'] = sym.zeros(self.n,1)
        self.i['vec'] = sym.zeros(self.n,1)
        for i,key in enumerate(self.var_names):
            self.i['vec'][i] = self.i[key+'_epsA']
            self.i['vec'][i] = self.i[key+'_epsB']
        
    def load_limit_cycle(self):
        
        self.LC['dat'] = []
        
        for key in self.var_names:
            self.LC['imp_'+key] = []
            self.LC['lam_'+key] = []
            
    
        file_does_not_exist = not(os.path.isfile(self.LC['dat_fname']))
        #print(os.path.isfile(self.LC['dat_fname']))
        if self.recompute_LC or file_does_not_exist:
            print('* Computing... LC data')
            
            # get limit cycle (LC) period
            sol,t_arr = self.generate_limit_cycle()
            
            # save LC data 
            np.savetxt(self.LC['dat_fname'],sol)
            np.savetxt(self.LC['t_fname'],t_arr)
            
        else:
            print('loading LC')
            sol = np.loadtxt(self.LC['dat_fname'])
            t_arr = np.loadtxt(self.LC['t_fname'])
                
        self.LC['dat'] = sol
        self.LC['t'] = t_arr
        
        # define basic variables
        self.T = self.LC['t'][-1]
        self.tLC = np.linspace(0,self.T,self.TN)#self.LC['t']
        self.omega = 2*np.pi/self.T

        print('* LC period = '+str(self.T))
            
        # Make LC data callable from inside sympy
        for i,key in enumerate(self.var_names):
            fn = interpb(self.LC['t'],self.LC['dat'][:,i],self.T)
            self.LC['imp_'+key] = imp_fn(key,fn)
            #self.LC['lam_'+key] = lambdify(self.t,self.LC['imp_'+key](self.t))
            self.LC['lam_'+key] = fn
            
            
        if False:
            #print('lc init',self.LC['dat'][0,:])
            #print('lc final',self.LC['dat'][-1,:])
            fig, axs = plt.subplots(nrows=self.n,ncols=1)
            print('LC init',end=', ')
            
            for i, ax in enumerate(axs.flat):
                key = self.var_names[i]
                ax.plot(self.tLC,self.LC['lam_'+key](self.tLC))
                print(self.LC['lam_'+key](0),end=', ')
            axs[0].set_title('LC')
            
            print()
            plt.tight_layout()
            plt.show(block=True)
            
        # coupling rules
        thA = self.thA
        thB = self.thB
            
        rule_dictA = {self.A_vars[i]:self.LC['imp_'+key](thA)
                      for i,key in enumerate(self.var_names)}
        rule_dictB = {self.B_vars[i]:self.LC['imp_'+key](thB)
                      for i,key in enumerate(self.var_names)}
        
        self.rule_LC_AB = {**rule_dictA,**rule_dictB}
        
    def generate_limit_cycle(self):
        
        tol = 1e-12
        T_init = 10.6
        eps = np.zeros(self.n) + 1e-2
        epstime = 1e-4
        dy = np.zeros(self.n+1)+10

        # rough init found using XPP
        init = np.array([-.64,0.71,0.25,0,T_init])
        
        #init = np.array([-.3,
        #                 .7619,
        #                 0.1463,
        #                 0,
        #                 T_init])
        
        
        # run for a while to settle close to limit cycle
        sol = solve_ivp(self.thal_rhs,[0,500],init[:-1],
                        method=self.method,dense_output=True,
                        rtol=1e-13,atol=1e-13)
        
        if False:
            fig, axs = plt.subplots(nrows=self.n,ncols=1)
                
            for i,ax in enumerate(axs):
                key = self.var_names[i]
                ax.plot(sol.t[:],sol.y.T[:,i],label=key)
                ax.legend()
                
            axs[0].set_title('long-time solution')
            plt.tight_layout()
            plt.show(block=True)
            time.sleep(.1)
            
        tn = len(sol.y.T)
        maxidx = np.argmax(sol.y.T[int(.2*tn):,0])+int(.2*tn)

        
        
            
        init = np.append(sol.y.T[maxidx,:],T_init)
        
        #init = np.array([-4.65e+01,8.77e-01,4.68e-04,T_init])
        #init = np.array([1.,1.,1.,1.,T_init])
        
        
        
        counter = 0
        while np.linalg.norm(dy) > tol:
            
            
            #eps = np.zeros(self.n)+1e-8
            #for i in range(self.n):
            #    eps[i] += np.amax(np.abs(sol.y.T[:,i]))*(1e-5)
            J = np.zeros((self.n+1,self.n+1))
            
            t = np.linspace(0,init[-1],self.TN)
            
            for p in range(self.n):
                pertp = np.zeros(self.n)
                pertm = np.zeros(self.n)
                
                
                pertp[p] = eps[p]
                pertm[p] = -eps[p]
                
                initp = init[:-1] + pertp
                initm = init[:-1] + pertm
                
                # get error in position estimate
                solp = solve_ivp(self.thal_rhs,[0,t[-1]],initp,
                                 method=self.method,dense_output=True,
                                 t_eval=t,
                                 rtol=self.rtol,atol=self.atol)
                
                solm = solve_ivp(self.thal_rhs,[0,t[-1]],initm,
                                 method=self.method,dense_output=True,
                                 t_eval=t,
                                 rtol=self.rtol,atol=self.atol)
            
            
                
                yp = solp.y.T
                ym = solm.y.T

                J[:-1,p] = (yp[-1,:]-ym[-1,:])/(2*eps[p])
                
                
            
            J[:-1,:-1] = J[:-1,:-1] - np.eye(self.n)
            
            
            tp = np.linspace(0,init[-1]+epstime,self.TN)
            tm = np.linspace(0,init[-1]-epstime,self.TN)
            
            # get error in time estimate
            solp = solve_ivp(self.thal_rhs,[0,tp[-1]],initp,
                             method=self.method,
                             rtol=self.rtol,atol=self.atol)
            
            solm = solve_ivp(self.thal_rhs,[0,tm[-1]],initm,
                             method=self.method,
                             rtol=self.rtol,atol=self.atol)
            
            yp = solp.y.T
            ym = solm.y.T
            
            J[:-1,-1] = (yp[-1,:]-ym[-1,:])/(2*epstime)
            
            J[-1,:] = np.append(self.thal_rhs(0,init[:-1]),0)
            #print(J)
            
            sol = solve_ivp(self.thal_rhs,[0,init[-1]],init[:-1],
                             method=self.method,
                             rtol=self.rtol,atol=self.atol)
            
            y_final = sol.y.T[-1,:]
            

            #print(np.dot(np.linalg.inv(J),J))
            b = np.append(init[:-1]-y_final,0)
            dy = np.dot(np.linalg.inv(J),b)
            init += dy
            
            print('LC rel. err =',np.linalg.norm(dy))
            
            
            if False:
                fig, axs = plt.subplots(nrows=self.n,ncols=1)
                
                for i,ax in enumerate(axs):
                    key = self.var_names[i]
                    ax.plot(sol.t,sol.y.T[:,i],label=key)
                    ax.legend()
                    
                axs[0].set_title('LC counter'+str(counter))
                plt.tight_layout()
                plt.show(block=True)
                time.sleep(.1)
                
            counter += 1
        

        # find index of peak voltage and initialize.
        peak_idx = np.argmax(sol.y.T[:,0])
        
        #init = np.zeros(5)
        #init[-1] = sol.t[-1]
        #init[:-1] = np.array([-0.048536698617817,
        #                      0.256223512263409,
        #                      0.229445856262051,
        #                      0.438912900900591])
        
        # run finalized limit cycle solution
        sol = solve_ivp(self.thal_rhs,[0,init[-1]],sol.y.T[peak_idx,:],
                        method=self.method,
                        t_eval=np.linspace(0,init[-1],self.TN),
                        rtol=self.rtol,atol=self.atol)
        
        #print('warning: lc init set by hand')
        #sol = solve_ivp(self.thal_rhs,[0,init[-1]],init[:-1],
        #                method='LSODA',
        #                t_eval=np.linspace(0,init[-1],self.TN),
        #                rtol=self.rtol,atol=self.atol)
            
        return sol.y.T,sol.t


    def load_monodromy(self):
        """
        if monodromy data exists, load. if DNE or recompute required, compute here.
        """
        
        
        if self.recompute_monodromy or not(os.path.isfile(self.monodromy_fname)):
            
            initm = copy.deepcopy(self.eye)
            r,c = np.shape(initm)
            init = np.reshape(initm,r*c)
            
            sol = solve_ivp(lib.monodromy2,[0,self.tLC[-1]],init,
                            args=(self,),t_eval=self.tLC,
                            method=self.method,
                            rtol=1e-10,atol=1e-10)
            
            self.sol = sol.y.T
            self.M = np.reshape(self.sol[-1,:],(r,c))
            np.savetxt(self.monodromy_fname,self.M)

        else:
            self.M = np.loadtxt(self.monodromy_fname)
        
        if False:
            fig, axs = plt.subplots(nrows=self.n,ncols=1,figsize=(10,10))
            sol = solve_ivp(self.mono1,[0,self.tLC[-1]],[0,0,0,1],
                            args=(self.jacLC,),t_eval=self.tLC,
                            method=self.method,dense_output=True,
                            rtol=self.rtol,atol=self.atol)
            
            for i,ax in enumerate(axs):
                ax.plot(self.tLC,sol.y.T[:,i])
                
            plt.tight_layout()
            plt.show(block=True)
        
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.M)
        
        # get smallest eigenvalue and associated eigenvector
        self.min_lam_idx = np.argsort(self.eigenvalues)[-2]
        #print(self.min_lam_idx)
        #print(self.eigenvalues[self.min_lam_idx])
        self.lam = self.eigenvalues[self.min_lam_idx]  # floquet mult.
        self.kappa = np.log(self.lam)/self.T  # floquet exponent
        
        if np.sum(self.eigenvectors[:,self.min_lam_idx]) < 0:
            self.eigenvectors[:,self.min_lam_idx] *= -1
        
        #print('eigenvalues',self.eigenvalues)
        #print('eiogenvectors',self.eigenvectors)
        
        #print(self.eigenvectors)
        
        # print floquet multipliers
        
        
        einv = np.linalg.inv(self.eigenvectors/2)
        #print('eig inverse',einv)
        
        idx = np.argsort(np.abs(self.eigenvalues-1))[0]
        #min_lam_idx2 = np.argsort(einv)[-2]
        
        
            
        self.g1_init = self.eigenvectors[:,self.min_lam_idx]/2.
        self.z0_init = einv[idx,:]
        self.i0_init = einv[self.min_lam_idx,:]

        #print('min idx for prc',idx,)
        
        #print('Monodromy',self.M)
        #print('eigenvectors',self.eigenvectors)
        
        #print('g1_init',self.g1_init)
        #print('z0_init',self.z0_init)
        #print('i0_init',self.i0_init)
        
        #print('Floquet Multiplier',self.lam)
        print('* Floquet Exponent kappa =',self.kappa)
        
        
        if False:
            fig, axs = plt.subplots(nrows=self.n,ncols=self.n,figsize=(10,10))
            
            for i in range(self.n):
                for j in range(self.n):
                    
                    axs[i,j].plot(self.tLC,self.sol[:,j+i*self.n])
                
            axs[0,0].set_title('monodromy')
            plt.tight_layout()
            plt.show(block=True)
            time.sleep(.1)
        
        
    def load_g_sym(self):
        # load het. functions h if they exist. otherwise generate.
        #self.rule_g0 = {sym.Indexed('gx',0):s(0),sym.Indexed('gy',0):s(0)}
        
        # create dict of gv0=0,gh0=0,etc for substitution later.
        self.rule_g0 = {sym.Indexed('g'+name,0):s(0) for name in self.var_names}
        
        for key in self.var_names:
            self.g['sym_'+key] = []
        #self.g_sym = {k: [] for k in self.var_names}
        
        # check that files exist
        val = 0
        for key in self.var_names:
            val += not(lib.files_exist(self.g['sym_fnames_'+key]))
        
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False
        
        if self.recompute_g_sym or files_do_not_exist:
            #print(self.recompute_g_sym,files_do_not_exist)
            print('* Computing... g sym')
            sym_collected = self.generate_g_sym()  # create symbolic derivative
            
            for i in range(self.miter):
                for key in self.var_names:
                    expr = sym_collected[key].coeff(self.psi,i)
        
        
                    self.g['sym_'+key].append(expr)
                    #print(self.g_sym_fnames[key][i])
                    dill.dump(self.g['sym_'+key][i],
                              open(self.g['sym_fnames_'+key][i],'wb'),
                              recurse=True)

                    
        else:
            for key in self.var_names:
                self.g['sym_'+key] = lib.load_dill(self.g['sym_fnames_'+key])
                
        """
        rule_tmp = {Indexed('g'+key,1):1 for key in self.var_names}
        rule_tmp3 = {Indexed('g'+key,2):2 for key in self.var_names}
        
        rule_tmp2 = {self.v:3,self.h:1.5,self.r:.1,self.w:1.2}
        rule_tmp = {**rule_tmp,**self.rule_par,**rule_tmp2,**rule_tmp3}
        print(rule_tmp)
        expr_temp = self.g['sym_w'][2]
        expr_temp = expr_temp.subs(rule_tmp)
        print(sym.N(expr_temp))
        lam_temp = lambdify(self.vars,expr_temp(*self.vars))
        
        print(lam_temp(1,1,1,1))
        """
        
    def generate_g_sym(self):
        """
        generate heterogeneous terms for the Floquet eigenfunctions g.
        
        purely symbolic.

        Returns
        -------
        list of symbolic heterogeneous terms in self.ghx_sym, self.ghy_sym.

        """
        
        # get the general expression for h before plugging in g.
        het = {key: sym.zeros(1,1) for key in self.var_names}
        
        for i in range(2,self.trunc_derivative+1):
            p = lib.kProd(i,self.dx_vec)
            #print(p)
            for j,key in enumerate(self.var_names):
                
                d = lib.vec(lib.df(self.thal_sym[j],self.x_vec,i))
                #print('j,d',j,d)
                #print('i,j',i,j,np.shape(p*d))
                #print()
                het[key] += (1/math.factorial(i)) * (p*d)
                #print((1/math.factorial(i)))
                
        #print(h)
        out = {}
        #  collect in psi.
        for key in self.var_names:
            #print()
            # remove small floating points
            tmp = het[key][0]
            tmp = sym.expand(tmp.subs(self.rule_d2g))
            #print('tmp1',key,type(tmp[0]),np.shape(tmp[0]))
            tmp = sym.collect(tmp.subs(self.rule_g0),self.psi)
            
            out[key] = tmp
            #print()
            #print(tmp)
            
        return out
    
    def load_g(self):
        """
        load all Floquet eigenfunctions g or recompute
        """
        
        self.g['dat'] = []
        
        for key in self.var_names:
            self.g['imp_'+key] = []
            self.g['lam_'+key] = []
        
        print('* Computing...', end=' ')
        for i in range(self.miter):
            print('g_'+str(i),end=', ')
            fname = self.g['dat_fnames'][i]
            #print('i,fname',i,fname)
            file_does_not_exist = not(os.path.exists(fname))
            if self.recompute_g or file_does_not_exist:
                
                rule = {}
                for key in self.var_names:
                    tmp = {sym.Indexed('g'+key,j):self.g['imp_'+key][j](self.t)
                           for j in range(i)}
                    #print(k,tmp)
                    rule.update(tmp)
                    
                rule = {**rule,**self.rule_LC,**self.rule_par}
                
                het_lams = {}
                # lambdify heterogeneous terms for use in integration
                for key in self.var_names:
                    lam = lambdify(self.t,self.g['sym_'+key][i].subs(rule))
                    
                    if i == 0 or i == 1:
                        y = np.zeros(self.TN)
                    else:
                        y = lam(self.tLC)
                        
                    het_lams[key] = interpb(self.tLC,y,self.T)
                        
                data = self.generate_g(i,het_lams)
                np.savetxt(self.g['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if False:
                fig, axs = plt.subplots(nrows=self.n,ncols=1)
                
                for j,ax in enumerate(axs):
                    key = self.var_names[j]
                    ax.plot(self.tLC,data[:,j],label=key)
                    ax.legend()
                    
                axs[0].set_title('g'+str(i))
                print('g'+str(i)+' init',data[0,:])
                plt.tight_layout()
                plt.show(block=True)
                time.sleep(.1)
                
                
            self.g['dat'].append(data)
            
            for j,key in enumerate(self.var_names):
                #print(len(self.tLC),len(data[:,j]))
                fn = interpb(self.tLC,data[:,j],self.T)
                imp = imp_fn('g'+key+'_'+str(i),self.fmod(fn))
                self.g['imp_'+key].append(imp)
                self.g['lam_'+key].append(fn)
                
        
        # replacement rules.
        thA = self.thA
        thB = self.thB
        
        self.rule_g = {}  # g function
        self.rule_g_AB = {}  # coupling
        for key in self.var_names:
            for i in range(self.miter):
                dictg = {sym.Indexed('g'+key,i):self.g['imp_'+key][i](self.t)}
                dictA = {Indexed('g'+key+'A',i):self.g['imp_'+key][i](thA)}
                dictB = {Indexed('g'+key+'B',i):self.g['imp_'+key][i](thB)}
                
                self.rule_g.update(dictg)
                self.rule_g_AB.update(dictA)
                self.rule_g_AB.update(dictB)
                
        print()
    
    def generate_g(self,k,het_lams):
        """
        generate Floquet eigenfunctions g
        
        uses Newtons method
        """
        # load kth expansion of g for k >= 0
        
        if k == 0:
            # g0 is 0. dot his to keep indexing simple.
            
            return np.zeros((self.TN,len(self.var_names)))
        
        
        
        #print('dg vec',k,self.dg(1,[1,1,1,1],k))
        

        if k == 1:
            # pick correct normalization
            #init = [0,self.g1_init[1],self.g1_init[2],self.g1_init[3]]
            init = copy.deepcopy(self.g1_init)
        else:
            init = np.zeros(self.n)
            
            # find intial condtion
        
        if k == 1:
            eps = 1e-5
            backwards = False
        else:
            eps = 1e-5
            
            if k == 3:
                backwards = False
            else:
                backwards = False
                
            init = lib.run_newton2(self,self.dg,init,k,het_lams,
                                  max_iter=100,eps=eps,
                                  rel_tol=self.rel_tol,rel_err=10,
                                  backwards=backwards)
        
        # get full solution
        
        if backwards:
            tLC = -self.tLC
            
        else:
            tLC = self.tLC
            
        sol = solve_ivp(self.dg,[0,tLC[-1]],
                        init,args=(k,het_lams),
                        t_eval=tLC,
                        method=self.method,dense_output=True,
                        rtol=self.rtol,atol=self.atol)
        
        if backwards:
            gu = sol.y.T[::-1,:]
            
        else:
            gu = sol.y.T
        
            
        return gu


    def load_het_sym(self):
        # load het. for z and i if they exist. otherwise generate.
        
        for key in self.var_names:
            self.het1['sym_'+key] = []
        #self.het1 = {'sym_'+k: [] for k in self.var_names}
        
        # check that files exist
        val = 0
        for key in self.var_names:
            val += not(lib.files_exist(self.het1['sym_fnames_'+key]))
        
        val += not(lib.files_exist([self.A_fname]))
        
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False
        
        if self.recompute_het_sym or files_do_not_exist:
            
            sym_collected = self.generate_het_sym()
            
            for i in range(self.miter):
                for key in self.var_names:
                    
                    expr = sym_collected[key][self.psi**i].subs(self.rule_g0)
                    self.het1['sym_'+key].append(expr)
                    #print('het1 key, i,expr', key, i,expr)
                    #print()
                    #print(self.g_sym_fnames[key][i])
                    dill.dump(self.het1['sym_'+key][i],
                              open(self.het1['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                
            # save matrix of a_i
            dill.dump(self.A,open(self.A_fname,'wb'),recurse=True)
            

        else:
            self.A, = lib.load_dill([self.A_fname])
            for key in self.var_names:
                self.het1['sym_'+key] = lib.load_dill(self.het1['sym_fnames_'+key])
        
        #lam = lambdify(self.t,self.het1['sym_'+key][1].subs(rule))
            
    def generate_het_sym(self):
        """
        Generate heterogeneous terms for integrating the Z_i and I_i terms.

        Returns
        -------
        None.

        """
        
        # get the general expression for h in z before plugging in g,z.
        
        # column vectors ax ay for use in matrix A = [ax ay]
        self.a = {k: Matrix([[0],[0],[0],[0]]) for k in self.var_names}
        
        #self.ax = Matrix([[0],[0]])
        #self.ay = Matrix([[0],[0]])
        
        for i in range(1,self.trunc_derivative+1):
            p1 = lib.kProd(i,self.dx_vec)
            p2 = kp(p1,sym.eye(self.n))

            for j,key in enumerate(self.var_names):
                
                d1 = lib.vec(lib.df(self.thal_sym[j],self.x_vec,i+1))
                #print((1/math.factorial(i)))
                self.a[key] += (1/math.factorial(i))*p2*d1
                
                
          
        self.A = sym.zeros(self.n,self.n)
        
        for i,key in enumerate(self.var_names):            
            self.A[:,i] = self.a[key]
        
        het = self.A*self.z['vec']
        
        # expand all terms
        out = {}
        for i,key in enumerate(self.var_names):
            het_key = sym.expand(het[i]).subs(self.rule_d2g)
            het_key = sym.collect(het_key,self.psi)
            het_key = sym.expand(het_key)
            het_key = sym.collect(het_key,self.psi,evaluate=False)
            #print(key,het_key)
            
            #print(i,key,het_key)
            #print(sym.apart(expr))
            #print(sym.collect(expr,self.psi,evaluate=False))
            #het_key = sym.collect(het_key,self.psi,evaluate=False)
            out[key] = het_key
            
        #het = {key: sym.expand(het[i]).subs(self.rule_d2g)
        #       for i,key in enumerate(self.var_names)}
        #self.hetx = sym.expand(het[0].subs([(self.dx,self.gx),(self.dy,self.gy)]))
        #self.hety = sym.expand(het[1].subs([(self.dx,self.gx),(self.dy,self.gy)]))
        
        # collect all psi terms into factors of pis^k
        #self.het1_collected = {k: sym.collect(het[k],self.psi,evaluate=False)
        #                        for k in self.var_names}
        
        return out
    
        
    def load_z(self):
        """
        load all PRCs z or recompute
        """
        
        self.z['dat'] = []
        
        for key in self.var_names:
            self.z['imp_'+key] = []
            self.z['lam_'+key] = []
            
        print('* Computing...', end=' ')
        for i in range(self.miter):
            print('z_'+str(i),end=', ')
            fname = self.z['dat_fnames'][i]
            file_does_not_exist = not(os.path.exists(fname))
            #print('z fname',fname)
            if self.recompute_z or file_does_not_exist:
                
                het_lams = self.interp_lam(i,self.z)
                
                data = self.generate_z(i,het_lams)
                np.savetxt(self.z['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if False:
                fig, axs = plt.subplots(nrows=self.n,ncols=1)
                
                for j,ax in enumerate(axs):
                    key = self.var_names[j]
                    ax.plot(self.tLC,data[:,j],label=key)
                    ax.legend()
                
                print('z'+str(i)+' init',data[0,:])
                axs[0].set_title('z'+str(i))
                plt.tight_layout()
                plt.show(block=True)
                time.sleep(.1)
                    
            self.z['dat'].append(data)
            
            for j,key in enumerate(self.var_names):
                fn = interpb(self.tLC,data[:,j],self.T)
                imp = imp_fn('z'+key+'_'+str(i),self.fmod(fn))
                self.z['imp_'+key].append(imp)
                self.z['lam_'+key].append(fn)
                
                #fn_temp = interpb(self.tLC,data[:,j],self.T)
                
                
                #imp_temp = imp_fn('z'+key+'_'+str(i),self.fmod(fn_temp))
                #lam_temp = lambdify(self.t,self.z['imp_'+key][i](self.t))
                #self.z['lam_'+key].append(lam_temp)
            

        
        print()
        # coupling
        thA = self.thA
        thB = self.thB
        
        self.rule_z_AB = {}
        for key in self.var_names:
            for i in range(self.miter):
                dictA = {Indexed('z'+key+'A',i):self.z['imp_'+key][i](thA)}
                dictB = {Indexed('z'+key+'B',i):self.z['imp_'+key][i](thB)}
                
                self.rule_z_AB.update(dictA)
                self.rule_z_AB.update(dictB)


        
        
    def generate_z(self,k,het_lams):
        
        if k == 0:
            init = copy.deepcopy(self.z0_init)
            eps = 1e-5
            #init = [-1.389, -1.077, 9.645, 0]
        else:
            init = np.zeros(self.n)
            eps = 1e-2
                
            init = lib.run_newton2(self,self.dz,init,k,het_lams,
                                  max_iter=50,eps=eps,
                                  rel_tol=self.rel_tol,rel_err=10,
                                  backwards=True)
            
        sol = solve_ivp(self.dz,[0,-self.tLC[-1]],
                        init,args=(k,het_lams),
                        method=self.method,dense_output=True,
                        t_eval=-self.tLC,
                        rtol=self.rtol,atol=self.atol)
            
        zu = sol.y.T[::-1]
        #zu = sol.y.T
        
        if k == 0:
            # normalize
            v0,h0,r0,w0 = [self.LC['lam_v'](0),
                           self.LC['lam_h'](0),
                           self.LC['lam_r'](0),
                           self.LC['lam_w'](0)]
            
            dLC = self.thal_rhs(0,[v0,h0,r0,w0])
            zu = self.omega*zu/(np.dot(dLC,zu[0,:]))
            
        return zu
    

    def load_i(self):
        """
        load all IRCs i or recomptue
        """
        
        self.i['dat'] = []
        
        for key in self.var_names:
            self.i['imp_'+key] = []
            self.i['lam_'+key] = []
        
        print('* Computing...', end=' ')
        for i in range(self.miter):
            print('i_'+str(i),end=', ')
            fname = self.i['dat_fnames'][i]
            file_does_not_exist = not(os.path.exists(fname))
            
            if self.recompute_i or file_does_not_exist:
                
                het_lams = self.interp_lam(i,self.i)
                
                data = self.generate_i(i,het_lams)
                np.savetxt(self.i['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if False:
                fig, axs = plt.subplots(nrows=self.n,ncols=1)
                
                for j,ax in enumerate(axs):
                    key = self.var_names[j]
                    ax.plot(self.tLC,data[:,j],label=key)
                    ax.legend()
                    
                print('i'+str(i)+' init',data[0,:])
                axs[0].set_title('i'+str(i))
                plt.tight_layout()
                plt.show(block=True)
                time.sleep(.1)
                
            self.i['dat'].append(data)
            
            for j,key in enumerate(self.var_names):
                fn = interpb(self.tLC,data[:,j],self.T)
                imp = imp_fn('i'+key+'_'+str(i),self.fmod(fn))
                self.i['imp_'+key].append(imp)
                self.i['lam_'+key].append(fn)
                
                #lam_temp = lambdify(self.t,self.i['imp_'+key][i](self.t))
                
            
               
        print()
        
        # coupling
        thA = self.thA
        thB = self.thB
        
        self.rule_i_AB = {}
        for key in self.var_names:
            for i in range(self.miter):
                dictA = {Indexed('i'+key+'A',i):self.i['imp_'+key][i](thA)}
                dictB = {Indexed('i'+key+'B',i):self.i['imp_'+key][i](thB)}
                
                self.rule_i_AB.update(dictA)
                self.rule_i_AB.update(dictB)
        
    
    
    def generate_i(self,k,het_lams):
        """
        i0 equation is stable in forwards time
        i1, i2, etc equations are stable in backwards time.

        """
        

        """
        # load kth expansion of g for k >= 1
        rule = {}
        for key in self.var_names:
            tmp = {sym.Indexed('z'+key,i):self.i['imp_'+key][i](self.t)
                   for i in range(k)}
            rule.update(tmp)
        
        rule = {**rule,**self.rule_g,**self.rule_LC,**self.rule_par}
        
        # lambdify heterogeneous terms for use in integration
        het_lams = {}
        for key in self.var_names:
            lam = lambdify(self.t,self.het1['sym_'+key][k].subs(rule))

            
            if k == 0:
                y = np.zeros(self.TN)
            else:
                y = lam(self.LC['t'])
            het_lams[key] = interpb(self.LC['t'],y)
            het_lams[key] = lam
       """ 
            
            
        if k == 0:
            init = copy.deepcopy(self.i0_init)
            eps = 1e-2
            exception=False
            #print(init)
            #sol = solve_ivp(self.di,[0,-self.tLC[-1]],init,
            #                args=(k,),
            #                t_eval=-self.tLC,
            #                method=self.method,dense_output=True,
            #                rtol=self.rtol,atol=self.atol)
            
            #iu = sol.y.T[::-1,:]
            
        else:
            
            #print('het i',het_lams['v'](1))
            init = np.zeros(self.n)
        
            if k == 1:
                exception = True
                eps = 1
            else:
                exception = False
                eps = 1e-1
                
                
            init = lib.run_newton2(self,self.di,init,k,het_lams,
                                   max_iter=50,rel_tol=self.rel_tol,
                                   rel_err=5,eps=eps,
                                   backwards=True,exception=exception)
        #t = -1.06881
        #print('i init',init)
        #print('myY',self.LC['lam_v'](t),self.LC['lam_h'](t))
        
        #print('di',self.di(t,[2.72731,0.896643,24.0568,0],k,het_lams))
        #print('jlc',self.jacLC(t).T)
        
        sol = solve_ivp(self.di,[0,-self.tLC[-1]],init,
                        args=(k,het_lams),
                        t_eval=-self.tLC,
                        method=self.method,dense_output=True,
                        rtol=self.rtol,atol=self.atol)
    
        iu = sol.y.T[::-1,:]
        #iu = sol.y.T
        
        if k == 0:
            
            # normalize
            c = np.dot(self.g1_init,iu[0,:])
            print('g1 init',self.g1_init)
            print('iu[0,:]',iu[0,:])
            print('i0 init',self.i0_init)
            print('constant dot',c)
            iu /= c
            #time.sleep(10)
    
        if k == 1:  # normalize
            
            F = self.thal_rhs(0,[self.LC['lam_v'](0),
                                 self.LC['lam_h'](0),
                                 self.LC['lam_r'](0),
                                 self.LC['lam_w'](0)])
            
            g1 = np.array([self.g['lam_v'][1](0),
                           self.g['lam_h'][1](0),
                           self.g['lam_r'][1](0),
                           self.g['lam_w'][1](0)])
            
            z0 = np.array([self.z['lam_v'][0](0),
                           self.z['lam_h'][0](0),
                           self.z['lam_r'][0](0),
                           self.z['lam_w'][0](0)])
            
            i0 = np.array([self.i['lam_v'][0](0),
                           self.i['lam_h'][0](0),
                           self.i['lam_r'][0](0),
                           self.i['lam_w'][0](0)])
            
            J = self.jacLC(0)
            i1 = iu[0,:]
            
            ijg = np.dot(i0,np.dot(J,g1))
            be = (self.kappa - ijg - np.dot(i1,F))/(np.dot(z0,F))
            
            print('actual',np.dot(F,i1))
            print('expect',np.dot(i0,np.dot(self.kappa*self.eye-J,g1)))
            print('canchg',z0)
            print('amtchg',np.dot(F,z0))
            print('mymult',be)
            print('i1 unnormalized init',i1)
            
            
            
            init = iu[0,:] + be*z0
            
            sol = solve_ivp(self.di,[0,-self.tLC[-1]],init,
                            args=(k,het_lams),
                            t_eval=-self.tLC,
                            method=self.method,dense_output=True)
            
            iu = sol.y.T[::-1]
            
        
        return iu

    def load_k_sym(self):
        
        """
        kA, kB contain the ith order terms of expanding the coupling fun.
        cA, cB contain the derivatives of the coupling fn.
        """
        # load het. functions h if they exist. otherwise generate.
        #self.rule_g0 = {sym.Indexed('gx',0):s(0),sym.Indexed('gy',0):s(0)}
        
        for key in self.var_names:
            self.kA['sym_'+key] = []
            self.kA['imp_'+key] = []
            
            self.kB['sym_'+key] = []
            self.kB['dat_'+key] = []
            
        # check that files exist
        val = 0
        for key in self.var_names:
            val += not(lib.files_exist(self.kA['sym_fnames_'+key]))
            val += not(lib.files_exist(self.kB['sym_fnames_'+key]))
            val += not(os.path.isfile(self.cA['sym_fname']))
            val += not(os.path.isfile(self.cB['sym_fname']))
        
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False
        
        if self.recompute_k_sym or files_do_not_exist:
            
            print('* Computing... K symbolic')
            self.cA, self.cB = self.generate_k_sym()
            
            self.cA['vec'] = sym.zeros(self.n,1)
            self.cB['vec'] = sym.zeros(self.n,1)
            
            for i,key in enumerate(self.var_names):
                self.cA['vec'][i] = self.cA[key]
                self.cB['vec'][i] = self.cB[key]
            
            # dump
            dill.dump(self.cA['vec'],open(self.cA['sym_fname'],'wb'),recurse=True)
            dill.dump(self.cB['vec'],open(self.cB['sym_fname'],'wb'),recurse=True)
            
            for key in self.var_names:
                collectedA = collect(expand(self.cA[key]),self.eps)
                collectedA = collect(expand(collectedA),
                                     self.eps,evaluate=False)
                
                collectedB = collect(expand(self.cB[key]),
                                     self.eps,evaluate=False)
                
                
                self.cA[key+'_col'] = collectedA
                self.cB[key+'_col'] = collectedB
                
                
                for i in range(self.miter):
                    
                    # save each order to list and dill.
                    if self.cA[key+'_col']:
                        eps_i_termA = self.cA[key+'_col'][self.eps**i]
                        eps_i_termB = self.cB[key+'_col'][self.eps**i]
                        
                    else:
                        eps_i_termA = 0
                        eps_i_termB = 0
                    
                    self.kA['sym_'+key].append(eps_i_termA)
                    self.kB['sym_'+key].append(eps_i_termB)
                    
                    dill.dump(self.kA['sym_'+key][i],
                              open(self.kA['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    
                    dill.dump(self.kB['sym_'+key][i],
                              open(self.kB['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    
                
        else:
            for key in self.var_names:
                self.kA['sym_'+key] = lib.load_dill(self.kA['sym_fnames_'+key])
                self.kB['sym_'+key] = lib.load_dill(self.kB['sym_fnames_'+key])
            
            self.cA['vec'] = lib.load_dill([self.cA['sym_fname']])[0]
            self.cB['vec'] = lib.load_dill([self.cB['sym_fname']])[0]
            
        #print(self.cA)
        #print(self.cB)
        
    def generate_k_sym(self):
        # generate terms involving the coupling term (see K in paper).
        
        # find K_i^{j,k}
        coupA = self.thal_coupling(self.A_pair,option='sym')
        coupB = self.thal_coupling(self.B_pair,option='sym')


        # get expansion for coupling term

        # 0 and 1st derivative
        for i,key in enumerate(self.var_names):
            self.cA[key] = coupA[i]
            self.cA[key] += lib.df(coupA[i],self.A_pair,1).dot(self.dA_pair)
            
            self.cB[key] = coupB[i]
            self.cB[key] += lib.df(coupB[i],self.B_pair,1).dot(self.dB_pair)
        
        # 2nd + derivative
        for i in range(2,self.trunc_derivative+1):
            # all x1,x2 are evaluated on limit cycle x=cos(t), y=sin(t)
            kA = lib.kProd(i,self.dA_pair)
            kB = lib.kProd(i,self.dB_pair)
            #print(i)
            
            for key in self.var_names:
                dA = lib.vec(lib.df(self.cA[key],self.A_pair,i))
                dB = lib.vec(lib.df(self.cB[key],self.B_pair,i))
                
                self.cA[key] += (1/math.factorial(i))*kA.dot(dA)
                self.cB[key] += (1/math.factorial(i))*kB.dot(dB)
        
                #print('* ',key,dA,self.cA[key])
                
                
        rule = {}
        for i,key in enumerate(self.var_names):
            rule.update({self.dA_vars[i]:self.g[key+'_epsA']})
            rule.update({self.dB_vars[i]:self.g[key+'_epsB']})
            
        
        for key in self.var_names:
            self.cA[key] = self.cA[key].subs(rule)
            self.cB[key] = self.cB[key].subs(rule)
            #print('self.cA[key]',self.cA[key])
        #print('self.ca',self.cA)
        return self.cA, self.cB
        
    
    def load_p_sym(self):
        """
        generate/load the het. terms for psi ODEs.
            
        to be solved using integrating factor meothod.
        
        pA['sym'][k] is the forcing function of order k
        """
        
        self.pA['sym'] = []
        #self.pB['sym'] = []
        
        if self.recompute_p_sym or not(lib.files_exist(self.pA['sym_fnames'])):
            
            print('* Computing... p symbolic')
            #ircA = self.generate_p_sym()
            ircA = self.eps*self.i['vec']*self.cA['vec'].T
        
            ircA = collect(expand(ircA[0]),self.eps)
            ircA = collect(expand(ircA),self.eps)
            
            for i in range(self.miter):
                # save each order to list and dill.
                eps_i_termA = ircA.coeff(self.eps,i)
                #eps_i_termA -= self.kap_sym*Indexed('pA',i)
                self.pA['sym'].append(eps_i_termA)

                dill.dump(self.pA['sym'][i],
                          open(self.pA['sym_fnames'][i],'wb'),recurse=True)

        else:
            self.pA['sym'] = lib.load_dill(self.pA['sym_fnames'])
        
        
    def generate_p_sym(self):
        
        # collect left and right hand terms
        #ircA = self.kap_sym*self.pA['expand']\
        #        + self.eps*self.i['vec'].dot(self.cA['vec'])
        
        pass

    def load_p(self):
        """
        generate/load the ODEs for psi.
        """
        
        # load all p or recompute or compute new.
        self.pA['dat'] = []
        self.pA['imp'] = []
        self.pA['lam'] = []
        
        #self.pA_data, self.pB_data, self.pA_imp, self.pB_imp = ([] for i in range(4))
        #self.pA_callable, self.pB_callable = ([] for i in range(2))

        # generate
        #if self.recompute_p or not(lib.files_exist(self.pA_fnames,self.pB_fnames)):
        
        print('* Computing...',end=' ')    
        for i,fname in enumerate(self.pA['dat_fnames']):
            print('p_'+str(i),end=', ')
            if self.recompute_p or not(os.path.isfile(fname)):
                
                
                pA_data = self.generate_p(i)
                
                np.savetxt(self.pA['dat_fnames'][i],pA_data)
                
            else:
                pA_data = np.loadtxt(fname)
            
            pA_interp = interp2d(self.A_array,self.B_array,
                                 pA_data,bounds_error=False,
                                 fill_value=None)
            
            pA_imp = implemented_function('pA_'+str(i),
                                          self.myFunMod2A(pA_interp))
            
            self.pA['dat'].append(pA_data)
            
            if i == 0:
                self.pA['imp'].append(implemented_function('pA_0', lambda x: 0))
                self.pA['lam'].append(0)
            
            else:
                self.pA['imp'].append(pA_imp)
                self.pA['lam'].append(pA_interp)
                        
        print()
        
        ta = self.thA
        tb = self.thB
        
        
        rule_pA = {sym.Indexed('pA',i):self.pA['imp'][i](ta,tb)
                       for i in range(self.miter)}
        rule_pB = {sym.Indexed('pB',i):self.pA['imp'][i](tb,ta)
                       for i in range(self.miter)}
        
        self.rule_p_AB = {**rule_pA,**rule_pB}
        
    def p_rhs(self,t,p,het,a,b):
        
        "test over solution over time for given ta,tb"
        
        return self.kappa*p + het(a+t,b+t)
    
    def generate_p(self,k):
        
        
        
        self.A_array,self.dxA = np.linspace(0,self.T,self.NA[k],retstep=True)
        self.B_array,self.dxB = np.linspace(0,self.T,self.NB[k],retstep=True)
        
        self.A_mg, self.B_mg = np.meshgrid(self.A_array,self.B_array)
        
        self.interval,self.ds = np.linspace(0,self.smax,self.Ns,retstep=True)
        
        ta = self.thA
        tb = self.thB
        
        if k == 0:
            #pA0 is 0 (no forcing function)
            return np.zeros((self.NB,self.NA))
        
        # put these implemented functions into the expansion
        ruleA = {sym.Indexed('pA',i):self.pA['imp'][i](ta,tb) for i in range(k)}
        ruleB = {sym.Indexed('pB',i):self.pA['imp'][i](tb,ta) for i in range(k)}
        
        
        rule = {**ruleA, **ruleB,
                **self.rule_g_AB,
                **self.rule_i_AB,
                **self.rule_LC_AB,
                **self.rule_par}
        
        ph_impA = self.pA['sym'][k].subs(rule)
        #print('phimpA',ph_impA)
        
        # this lambidfy calls symbolic functions. slow.
        # convert lamdify to data and call linear interpolation on that.
        # then function call is same speed independent of order.
        lam_hetA = lambdify([ta,tb],ph_impA)
        lam_hetA_old = lam_hetA
        
        
        lam_hetA_data = np.zeros((self.NB,self.NA))
        
        for i in range(self.NA):
            ta2 = self.A_array[i]*np.ones_like(self.B_array)
            tb2 = self.B_array
            lam_hetA_data[:,i] = lam_hetA(ta2,tb2)
        
        het_interp = interp2d(self.A_array,self.B_array,
                              lam_hetA_data,bounds_error=False,
                              fill_value=None)
        
        pA_imp = implemented_function('temp',self.myFunMod2A(het_interp))
        lam_hetA = lambdify([ta,tb],pA_imp(ta,tb))
        
        if False:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.imshow(lam_hetA_old(self.A_mg,self.B_mg))
            ax2.imshow(het_interp(self.A_array,self.B_array))
                       
            plt.show(block=True)
        
        pA_data = np.zeros((self.NB,self.NA))
        
        r,c = np.shape(self.A_mg)
        a = np.reshape(self.A_mg,(r*c,))
        b = np.reshape(self.B_mg,(r*c,))
        
        for i in range(len(self.interval)):
            print('i in p'+str(k)+' function',i)
            s = self.interval[i]
            
            het = np.reshape(lam_hetA(a-s,b-s),(r,c))

            pA_data += np.exp(self.kappa*s)*het
            
        pA_data *= self.ds
        
        """
        # no choice but to double loop because of interp2d.
        for i in range(self.NA):
            print('i in constructing p function',i)
            
            
            for j in range(self.NB):
                
                #print('i,j in p',i,j)
                
                s = self.interval
                a, b = self.A_array[i], self.B_array[j]
                
                intA = np.exp(self.kappa*s)*lam_hetA(a-s,b-s)
                #intB = np.exp(self.kappa*s)*lam_hetB(b-s,a-s)
                
                if False and (i % 10 == 0) and (j % 10 == 0):
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.plot(self.interval,intA)
                    ax.set_title('inA_'+str(k)+' '+str(i)+','+str(j))
                    plt.show(block=True)
                    
                    s2 = np.linspace(0,600,2*len(self.interval))
                    int_temp = np.exp(self.kappa*s2)*lam_hetA(a-s2,b-s2)
                    print('2x interval value',
                          np.sum(int_temp)*self.ds)
                    print('integral value',np.sum(intA)*self.ds)
                    print('half interval value',
                          np.sum(intA[:int(len(self.interval)/2)])*self.ds)
                    
                
                pA_data[j,i] = np.sum(intA)*self.ds
                #pB_data[i,j,:] = intB
        """
        
        if False:
            # solve ODE version
            i = 10
            j = 21
            a = self.A_array[i]
            b = self.B_array[j]
            
            
            sol = solve_ivp(self.p_rhs,[0,500],[0],args=(lam_hetA,a,b),
                            method=self.method,dense_output=True,
                            rtol=1e-6,atol=1e-6)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sol.t,sol.y.T[:,0])
            ax.plot(sol.t,np.convolve(sol.y.T[:,0],np.zeros(100),mode='full'))
            
            plt.show(block=True)
            
            
            
            print('pa data, a, b, i, j',pA_data[j,i],a,b,i,j,sol.y.T[-1])
        
            time.sleep(60)
        
        # integrate
        #pA_data = np.sum(pA_data,axis=-1)*self.ds
        #pB_data = np.sum(pB_data,axis=-1)*self.ds
        
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.matshow(pA_data,cmap='viridis')
            
            ax.set_ylabel('A')
            ax.set_xlabel('B')
            ax.set_title('pA sum'+str(k))
            plt.show(block=True)
            plt.close()
        
        return pA_data
        
        
    def load_h_sym(self):
        # symbolic h terms
        
        self.hodd['sym'] = []
        
        if self.recompute_h_sym or \
            not(lib.files_exist(self.hodd['sym_fnames'])):
            
            print('* Computing... H symbolic')
            
            z_rule = {}
            for key in self.var_names:
                for i in range(self.miter):
                    z_rule.update({Indexed('z'+key,i):Indexed('z'+key+'A',i)})

            z_rule.update({self.psi:self.pA['expand']})
            z = self.z['vec'].subs(z_rule)
            coupling = self.cA['vec'].T
            dotproduct = coupling*z
            
            #print(dotproduct)
            #print(dotproduct[0])
            
            collected = collect(expand(dotproduct[0]),self.eps)
            self.h_collected = collect(expand(collected),self.eps)
            
            for i in range(self.miter):
                collected = self.h_collected.coeff(self.eps,i)
                #print(collected)
                self.hodd['sym'].append(collected)
                dill.dump(self.hodd['sym'][i],
                          open(self.hodd['sym_fnames'][i],'wb'),recurse=True)
                
        else:
            self.hodd['sym'] = lib.load_dill(self.hodd['sym_fnames'])
            
            
    def load_h(self):
        
        self.hodd['lam'] = []
        self.hodd['dat'] = []
        
        #self.i_data, self.ix_imp, self.iy_imp = ([] for i in range(3))
        #self.ix_callable, self.iy_callable = ([] for i in range(2))

        

        if self.recompute_h or not(lib.files_exist(self.hodd['dat_fnames'])):
            
            print('* Computing...',end=' ')
            rule = {**self.rule_p_AB,
                    **self.rule_g_AB,
                    **self.rule_z_AB,
                    **self.rule_LC_AB,
                    **self.rule_par}
            
            for i in range(self.miter):
                
                collected = self.hodd['sym'][i].subs(rule)
                
                #print('i,col',i,collected)
                
                ta = self.thA
                tb = self.thB
                
                h_lam = sym.lambdify([ta,tb],collected)
                self.hodd['lam'].append(h_lam)
                
            
            for k in range(self.miter):
                print('h_'+str(k),end=', ')
                data = self.generate_h_odd(k)
                self.hodd['dat'].append(data)
                np.savetxt(self.hodd['dat_fnames'][k],data)
            print()
            
            
        else:
            
            for k in range(self.miter):
                self.hodd['dat'].append(np.loadtxt(self.hodd['dat_fnames'][k]))
    
        """
        i = 0
        v = self.LC['dat_v']
        w = self.LC['dat_w']
        z0 = self.z['dat'][0][:,0]
        
        from numpy.fft import fft,ifft
        
        y = ifft(fft(v*z0)*fft(w[::-1]))
        
        print('h0',self.hodd['sym'][0])
        #y = convolve1d(v*z0,w,mode='wrap')
        print(y[::-1]-y)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(y[::-1]-y)
        plt.show(block=True)
        print('PLOT H')
        """
    
    def generate_h_odd(self,k):
        """
        interaction functions
        
        note to self: see nb page 130 for notes on indexing in sums.
        need to sum over to index N-1 out of size N to avoid
        double counting boundaries in mod operator.
        """
        
        h_mg = np.zeros((self.NB,self.NA))
        
        
        
        for j in range(self.NB):
            t = self.A_array
            eta = self.B_array[j]
            
            h_mg[j,:] = self.hodd['lam'][k](t,t+eta)
        
        # for i in range(self.NA):
        #     for j in range(self.NB):
        #         t = self.A_array[i]
        #         eta = self.B_array[j]
                
        #         h_mg[j,i] = self.h_lams[k](t,t+eta)
                
        # sum along axis to get final form
        h = np.sum(h_mg,axis=1)*self.dxA/self.T
        
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(h)
            ax.set_title('h non-odd'+str(k))
            plt.show(block=True)
        
        #print(h)
        #hodd = h
        hodd = (h[::-1]-h)
        
        return hodd
            
    def dg(self,t,z,order,het_lams):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        #z[0] *= 100
        #z[2] /= 100
        
        #jac = self.jacLC(t)*(order > 0)
        
        LC_vec = np.array([self.LC['lam_v'](t),
                           self.LC['lam_h'](t),
                           self.LC['lam_r'](t),
                           self.LC['lam_w'](t)])
        jac = self.numerical_jac(self.thal_rhs,LC_vec)*(order > 0)
        
        hom = np.dot(jac-order*self.kappa*self.eye,z)
        het = np.array([het_lams['v'](t),het_lams['h'](t),
                        het_lams['r'](t),het_lams['w'](t)])
        
        #if int(t*self.TN/self.tLC[-1])%100 == 0:
        #    print(hom,het,t,order)
    
        out = hom + het
    
        #out[0] /= 100
        #out[2] *= 100
    
    
    
        return out
    
    def dz(self,t,z,order,het_lams):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = -np.dot(self.jacLC(t).T+order*self.kappa*self.eye,z)
        
        
        het = -np.array([het_lams['v'](t),
                         het_lams['h'](t),
                         het_lams['r'](t),
                         het_lams['w'](t)])
        
        out = hom + het
        
        return out
    
    def di(self,t,z,order,het_lams):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        #z[0] *= 100
        #z[2] /= 100
        hom = -np.dot(self.jacLC(t).T+(order-1)*self.kappa*self.eye,z)
        
        het = -np.array([het_lams['v'](t),
                         het_lams['h'](t),
                         het_lams['r'](t),
                         het_lams['w'](t)])
        #het = -np.array([hetx(t),hety(t)])
        out = hom + het
        
        #out[0] /= 100
        #out[2] *= 100
        #print(int(t*self.TN/self.T)%10)
        
        if order == 1 and int(t*self.TN/self.T) % 100 == 0 and False:
            print(t,z,out,int(t*self.TN/self.T),het)
        
        return out
    
    def interp_lam(self,k,fn_dict):
        """
        it is too slow to call individual interpolated functions
        in the symbolic heterogeneous terms.
        soince the heterogeneous terms only depend on t, just make
        and interpolated version and use that instead so only 1 function
        is called for the het. terms per iteration in numerical iteration.
        """
        # lambdify heterogeneous terms for use in integration
        # create lambdified heterogeneous term and interpolate
        # load kth expansion of g for k >= 1
        rule = {}
        for key in self.var_names:
            tmp = {sym.Indexed('z'+key,i):fn_dict['imp_'+key][i](self.t)
                   for i in range(k)}
            #print(k,key,len(self.z['imp_'+key]))
            rule.update(tmp)
        
        rule = {**rule,**self.rule_g,**self.rule_LC,**self.rule_par}
        
        
        het_lams = {}
        for i,key in enumerate(self.var_names):
            lam = lambdify(self.t,self.het1['sym_'+key][k].subs(rule))
            #print('lam2',lam(self.LC['t']))
            
            t = np.linspace(0,self.T,self.TN)
            if k == 0:
                y = np.zeros(len(t))
            else:
                y = lam(t)
            het_lams[key] = interpb(t,y,self.T)
            #het_lams[key] = lam
            
              
        if False and k > 0:
            fig, axs = plt.subplots(nrows=self.n,ncols=1)
            for i,key in enumerate(self.var_names):
                print('k',k,key)                
                axs[i].plot(self.tLC*2,het_lams[key](self.tLC*2))
            
            axs[0].set_title('lam dict')
            plt.tight_layout()
            plt.show(block=True)
            
        return het_lams
    
    
    def fmod(self,fn):
        """
        fn has mod built-in
        
        input function-like. usually interp1d object
        
        needed to keep lambda input variable unique to fn.
        
        otherwise lambda will use the same input variable for 
        all lambda functions.
        """
        return lambda x=self.t: fn(x)
    

        
    def bispeu(self,fn,x,y):
        """
        silly workaround
        https://stackoverflow.com/questions/47087109/...
        evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
        """
        return si.dfitpack.bispeu(fn.tck[0], fn.tck[1],
                                  fn.tck[2], fn.tck[3],
                                  fn.tck[4], x, y)[0]
        
    def myFunMod2A(self,fn):
        """
        same as above but for 2 variable function for use with interp2d 
        function only.
        
        fn: must be interp2d function object
        xA and xB must have same 1d array sizes. f(float,array) wont work.
        
        """
        # need bispeu to allow for 1d array inputs.
        ta = self.thA
        tb = self.thB
        T = self.T
        
        return lambda xA=ta,xB=tb:self.bispeu(fn,np.mod(xA,T),np.mod(xB,T))
    
    def fmod2A(self,fn):
        """
        same as above but for 2 variable function for use with interp2d 
        function only.
        
        fn: must be interp2d function object
        xA and xB must have same 1d array sizes. f(float,array) wont work.
        
        """
        # need bispeu to allow for 1d array inputs.
        
        return lambda xA=self.thA,xB=self.thB: fn(np.mod(xA,self.T),
                                                  np.mod(xB,self.T))
    
    def myFunMod2B(self,fn):
        """
        same as above but for 2 variable function.
        """
        return lambda xB=self.thB,xA=self.thA: self.bispeu(fn,
                                                           np.mod(xB,self.T),
                                                           np.mod(xA,self.T))
    
    def myFunMod3(self,fn):
        """
        same as above but for 2 variable function.
        """
        return lambda x=self.s,xA=self.tA,xB=self.tB: fn(np.mod(xA-x,self.T),
                                                         np.mod(xB-x,self.T))



def main():
    
    a = Thalamic(recompute_LC=False,
                 recompute_monodromy=False,
                 recompute_g_sym=False,
                 recompute_g=False,
                 recompute_het_sym=False,
                 recompute_z=False,
                 recompute_i=False,
                 recompute_k_sym=False,
                 recompute_p_sym=False,
                 recompute_p=True,
                 recompute_h_sym=False,
                 recompute_h=True,
                 trunc_order=4,
                 trunc_derivative=4,
                 TN=2000,
                 load_all=True)

    """
    for i in range(a.miter):
        lib.plot(a,'g'+str(i))
    
    
    for i in range(a.miter):
        lib.plot(a,'z'+str(i))
    
    for i in range(a.miter):
        lib.plot(a,'i'+str(i))
    """
    for i in range(a.miter):
        lib.plot(a,'pA'+str(i))
    
    
    
    for i in range(a.miter):
        lib.plot(a,'hodd'+str(i))
    #lib.plot(a,'hodd1')
    
    
    """
    lib.plot(a,'surface_z')
    lib.plot(a,'surface_i')
    """
    
    """
    # check total hodd
    ve = .5
    h = 0
    for i in range(4):
        h += ve**(i+1)*a.h_odd_data[i]
        
        
    # get zeros
    idxs = np.arange(len(a.A_array))
    crossing_idx = np.where(np.diff(np.signbit(h)))[0]  # (np.diff(np.sign(h)) != 0)
    
    print(np.sign(h))
    print(crossing_idx)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(h)
    ax.plot([0,len(h)],[0,0],color='gray')
    ax.scatter(idxs[crossing_idx],np.zeros(len(idxs[crossing_idx])),color='red')
    
    """
    
    plt.show(block=True)
    
    
if __name__ == "__main__":
    
    import cProfile
    import re
    cProfile.runctx('main()',globals(),locals(),'profile.pstats')
    #cProfile.runctx('main()',globals(),locals())

    #main()
