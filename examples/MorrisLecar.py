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

def round2zero(m, e=1e-20):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if (isinstance(m[i,j], sym.Float) and m[i,j] < e):
                m[i,j] = 0

# user-defined
import MatchingLib as lib
from interp_basic import interp_basic as interpb
from interp2d_basic import interp2d_basic as interp2db
from lam_vec import lam_vec
import SymLib as slib

import inspect
import time
import os
import math
#import time
import dill
import copy
import matplotlib
import tqdm

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
from pathos.pools import _ProcessPool
#from sympy.interactive import init_printing

imp_fn = implemented_function

#from interpolate import interp1d
#from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import interp2d
from scipy.integrate import solve_ivp, quad

matplotlib.rcParams.update({'figure.max_open_warning': 0})


class MorrisLecar(object):
    """
    Thalamic model from RSTA 2019
    Requires sympy, numpy, matplotlib.
    """
    
    def __init__(self,**kwargs):

        """
        recompute_g_sym : recompute heterogenous terms for Floquet e.funs g
        
        """

        defaults = {
            'phi_val':0.333,
            #'ib_val':.2,
            'ib_val':0.24,
            'v1_val':-.01,
            'v2_val':.15,
            'v3_val':.1,
            'v4_val':.145,
            'vca_val':1,
            'vk_val':-.7,
            'gca_val':1,
            'gL_val':.5,
            'gk_val':2,
            'vL_val':-.5,
            'vth_val':.05,
            'vshp_val':.05,
            'c_val':1,
            'beta_val':4,
            'esyn_val':-.64, # see lecsimp2.ode

            'trunc_order':3,
            'trunc_derivative':2,
            
            'TN':20000,
            'dir':'dat',
            
            'recompute_LC':False,
            'recompute_monodromy':True,
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
        self.var_names = ['v','w','q']
        self.dim = len(self.var_names)
        
        # misc variables
        self.miter = self.trunc_order+1

        # Symbolic variables and functions
        self.eye = np.identity(self.dim)
        self.psi, self.eps, self.kap_sym = sym.symbols('psi eps kap_sym')
        
        self.rtol = 1e-11
        self.atol = 1e-11
        self.method = 'LSODA'
        self.rel_tol = 1e-9
        
         # for coupling computation. ctrl+F to see usage.
        self.NA = np.array([1000,1000,1000,1000,1000,1000,1000,1000])
        self.NB = self.NA # do not change!!!!
        self.Ns = self.NA # do not change!!!!
        self.smax = np.array([1,1,1,1,1,1,1,1])
        self.p_iter = np.array([25,25,25,25,25,25,25,25])
        
        #self.x1,self.x2,self.x3,self.t = symbols('x1 x2,x3, t')
        #self.x,self.y,self.z,self.t = symbols('x1 x2,x3, t')
        #self.f_list, self.x_vars, self.y_vars, self.z_vars = ([] for i in range(4))
        
        # parameters
        self.c = symbols('c')
        self.phi, self.ib = symbols('phi ib')
        self.v1, self.v2 = symbols('v1 v2')
        self.v3, self.v4 = symbols('v3 v4')
        self.vca, self.vk = symbols('vca vk')
        self.gca, self.gL = symbols('gca gL')
        self.gk, self.vL = symbols('gk vL')
        self.vth, self.vshp = symbols('vth vshp')
        self.esyn, self.beta = symbols('esyn beta')
        
        
        self.rule_par = {self.c:self.c_val,
                         self.phi:self.phi_val,
                         self.ib:self.ib_val,
                         self.v1:self.v1_val,
                         self.v2:self.v2_val,
                         self.v3:self.v3_val,
                         self.v4:self.v4_val,
                         self.vca:self.vca_val,
                         self.vk:self.vk_val,
                         self.gca:self.gca_val,
                         self.gL:self.gL_val,
                         self.gk:self.gk_val,
                         self.vL:self.vL_val,
                         self.vth:self.vth_val,
                         self.vshp:self.vshp_val,
                         self.beta:self.beta_val,
                         self.esyn:self.esyn_val
                         }
        

        # single-oscillator variables
        
        self.v, self.w, self.q, self.t, self.s = symbols('v w q t s')
        self.tA, self.tB, = symbols('tA tB')
        self.dv, self.dw, self.dq = symbols('dv dw dq')
        
        # coupling variables
        self.thA, self.psiA, self.thB, self.psiB = symbols('thA psiA thB psiB')
        
        self.vA, self.wA, self.qA = symbols('vA wA qA')
        self.vB, self.wB, self.qB = symbols('vB wB qB')

        self.dvA, self.dwA, self.dqA = symbols('dvA dwA dqA')
        self.dvB, self.dwB, self.dqB = symbols('dvB dwB dqB')

        self.vars = [self.v,self.w,self.q]
        self.A_vars = [self.vA,self.wA,self.qA]
        self.dA_vars = [self.dvA,self.dwA,self.dqA]
        
        self.B_vars = [self.vB,self.wB,self.qB]
        self.dB_vars = [self.dvB,self.dwB,self.dqB]
        
        self.A_pair = sym.zeros(1,2*self.dim)
        self.A_pair[0,:self.dim] = [self.A_vars]
        self.A_pair[0,self.dim:] = [self.B_vars]
        
        self.dA_pair = sym.zeros(1,2*self.dim)
        self.dA_pair[0,:self.dim] = [self.dA_vars]
        self.dA_pair[0,self.dim:] = [self.dB_vars]
        
        self.B_pair = sym.zeros(1,2*self.dim)
        self.B_pair[0,:self.dim] = [self.B_vars]
        self.B_pair[0,self.dim:] = [self.A_vars]
        
        self.dB_pair = sym.zeros(1,2*self.dim)
        self.dB_pair[0,:self.dim] = [self.dB_vars]
        self.dB_pair[0,self.dim:] = [self.dA_vars]
        
        
        self.dx_vec = Matrix([[self.dv,self.dw,self.dq]])
        self.x_vec = Matrix([[self.v],[self.w],[self.q]])
        
        # function dicts
        # individual functions
        self.LC = {}
        self.g = {}
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
        self.dir = 'ml_dat/'
        
        if (not os.path.exists(self.dir)):
            os.makedirs(self.dir)
        
        if self.ib_val != 0.2:
            lib.generate_fnames(self,model_pars='_iapp='+str(self.ib))
        else:
            lib.generate_fnames(self,model_pars='')
            
        #self.generate_fnames()
        
        # make rhs callable
        self.rhs_sym = self.rhs(0,self.vars,option='sym')
        
        self.load_limit_cycle()
        
        slib.load_jac_sym(self)
        
        slib.generate_expansions(self)
        slib.generate_coupling_expansions(self)
        slib.load_jac_sym(self)
        
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
        
    def rhs(self,t,z,option='val'):
        """
        right-hand side of the equation of interest. thalamic neural model.
        """
        
        v, w, q = z
        
        
        if option == 'val':
            phi = self.phi_val
            ib = self.ib_val
            v1 = self.v1_val
            v2 = self.v2_val
            
            v3 = self.v3_val
            v4 = self.v4_val
            vca = self.vca_val
            vk = self.vk_val
            
            gca = self.gca_val
            gL = self.gL_val
            gk = self.gk_val
            vL = self.vL_val
            vth= self.vth_val
            vshp = self.vshp_val
            
            c = self.c_val
            beta = self.beta_val
            exp = np.exp
            tanh = np.tanh
            cosh = np.cosh

        elif option == 'sym':
            phi = self.phi
            ib = self.ib
            v1 = self.v1
            v2 = self.v2
            
            v3 = self.v3
            v4 = self.v4
            vca = self.vca
            vk = self.vk
            
            gca = self.gca
            gL = self.gL
            gk = self.gk
            vL = self.vL
            vth= self.vth
            vshp = self.vshp
            
            c = self.c
            beta = self.beta
            exp = sym.exp
            tanh = sym.tanh
            cosh = sym.cosh
            
        else:
            raise ValueError('Unrecognized option',option)
            
        minf = 0.5*(1+tanh((v-v1)/v2))
        winf = 0.5*(1+tanh((v-v3)/v4))
        lamw = cosh((v-v3)/(2*v4))
        
        dv = (-gca*minf*(v-vca)-gk*w*(v-vk)-gL*(v-vL)+ib)/c
        dw = phi*lamw*(winf-w)
        dq = 1/(1+exp(-(v-vth)/vshp))-q*beta
        
        if option == 'val':
            return np.array([dv,dw,dq])
            #return np.array([dv,dw,dq])
            #return np.array([dv,dh,dr])
        else:
            return Matrix([dv,dw,dq])
            #return Matrix([dv,dw,dq])
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
        
    def coupling(self,vars_pair,option='val'):
        vA, wA, qA, vA, wB, qB = vars_pair

        if option == 'val':
            return np.array([qB*(vA-self.esyn_val),0,0,0])/self.c_val
        else:
            return Matrix([qB*(vA-self.esyn),0,0,0])/self.c

    
        
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
        lam_list = []
        for i,key in enumerate(self.var_names):
            fn = interpb(self.LC['t'],self.LC['dat'][:,i],self.T)
            self.LC['imp_'+key] = imp_fn(key,self.fmod(fn))
            self.LC['lam_'+key] = lambdify(self.t,self.LC['imp_'+key](self.t))
            lam_list.append(self.LC['lam_'+key])
            
        self.LC_vec = lam_vec(lam_list)
            
        if True:
            #print('lc init',self.LC['dat'][0,:])
            #print('lc final',self.LC['dat'][-1,:])
            fig, axs = plt.subplots(nrows=self.dim,ncols=1)
            print('LC init',end=', ')
            
            for i, ax in enumerate(axs.flat):
                key = self.var_names[i]
                ax.plot(self.tLC,self.LC['lam_'+key](self.tLC))
                print(self.LC['lam_'+key](0),end=', ')
            axs[0].set_title('LC')
            
            print()
            plt.tight_layout()
            plt.show(block=True)
    
        # single oscillator rules
        self.rule_LC = {}
        for i,key in enumerate(self.var_names):
            self.rule_LC.update({self.vars[i]:self.LC['imp_'+key](self.t)})
    
    
        # coupling rules
        thA = self.thA
        thB = self.thB
            
        rule_dictA = {self.A_vars[i]:self.LC['imp_'+key](thA)
                      for i,key in enumerate(self.var_names)}
        rule_dictB = {self.B_vars[i]:self.LC['imp_'+key](thB)
                      for i,key in enumerate(self.var_names)}
        
        self.rule_LC_AB = {**rule_dictA,**rule_dictB}
        
    def generate_limit_cycle(self):
        
        tol = 1e-13
        T_init = 7
        eps = np.zeros(self.dim) + 1e-4
        epstime = 1e-4
        dy = np.zeros(self.dim+1)+10

        # rough init found using XPP
        #init = np.array([-.64,0.71,0.25,0,T_init])
        init = np.array([0,0,0,T_init])
        
        #init = np.array([-.3,
        #                 .7619,
        #                 0.1463,
        #                 0,
        #                 T_init])
        
        
        # run for a while to settle close to limit cycle
        sol = solve_ivp(self.rhs,[0,500],init[:-1],
                        method=self.method,dense_output=True,
                        rtol=1e-13,atol=1e-13)
        
        if True:
            fig, axs = plt.subplots(nrows=self.dim,ncols=1)
                
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
            
            
            #eps = np.zeros(self.dim)+1e-8
            #for i in range(self.dim):
            #    eps[i] += np.amax(np.abs(sol.y.T[:,i]))*(1e-5)
            J = np.zeros((self.dim+1,self.dim+1))
            
            t = np.linspace(0,init[-1],self.TN)
            
            for p in range(self.dim):
                pertp = np.zeros(self.dim)
                pertm = np.zeros(self.dim)
                
                
                pertp[p] = eps[p]
                pertm[p] = -eps[p]
                
                initp = init[:-1] + pertp
                initm = init[:-1] + pertm
                
                # get error in position estimate
                solp = solve_ivp(self.rhs,[0,t[-1]],initp,
                                 method=self.method,dense_output=True,
                                 t_eval=t,
                                 rtol=self.rtol,atol=self.atol)
                
                solm = solve_ivp(self.rhs,[0,t[-1]],initm,
                                 method=self.method,dense_output=True,
                                 t_eval=t,
                                 rtol=self.rtol,atol=self.atol)
            
            
                
                yp = solp.y.T
                ym = solm.y.T

                J[:-1,p] = (yp[-1,:]-ym[-1,:])/(2*eps[p])
                
                
            
            J[:-1,:-1] = J[:-1,:-1] - np.eye(self.dim)
            
            
            tp = np.linspace(0,init[-1]+epstime,self.TN)
            tm = np.linspace(0,init[-1]-epstime,self.TN)
            
            # get error in time estimate
            solp = solve_ivp(self.rhs,[0,tp[-1]],initp,
                             method=self.method,
                             rtol=self.rtol,atol=self.atol)
            
            solm = solve_ivp(self.rhs,[0,tm[-1]],initm,
                             method=self.method,
                             rtol=self.rtol,atol=self.atol)
            
            yp = solp.y.T
            ym = solm.y.T
            
            J[:-1,-1] = (yp[-1,:]-ym[-1,:])/(2*epstime)
            
            J[-1,:] = np.append(self.rhs(0,init[:-1]),0)
            #print(J)
            
            sol = solve_ivp(self.rhs,[0,init[-1]],init[:-1],
                             method=self.method,
                             rtol=self.rtol,atol=self.atol)
            
            y_final = sol.y.T[-1,:]
            

            #print(np.dot(np.linalg.inv(J),J))
            b = np.append(init[:-1]-y_final,0)
            dy = np.dot(np.linalg.inv(J),b)
            init += dy
            
            print('LC rel. err =',np.linalg.norm(dy))
            
            
            if False:
                fig, axs = plt.subplots(nrows=self.dim,ncols=1)
                
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
        sol = solve_ivp(self.rhs,[0,init[-1]],sol.y.T[peak_idx,:],
                        method=self.method,
                        t_eval=np.linspace(0,init[-1],self.TN),
                        rtol=self.rtol,atol=self.atol)
        
        #print('warning: lc init set by hand')
        #sol = solve_ivp(self.rhs,[0,init[-1]],init[:-1],
        #                method='LSODA',
        #                t_eval=np.linspace(0,init[-1],self.TN),
        #                rtol=self.rtol,atol=self.atol)
            
        return sol.y.T,sol.t


    def load_monodromy(self):
        """
        if monodromy data exists, load. if DNE or recompute required, compute here.
        """
        
        
        if self.recompute_monodromy or \
            not(os.path.isfile(self.monodromy_fname)):
            
            initm = copy.deepcopy(self.eye)
            r,c = np.shape(initm)
            init = np.reshape(initm,r*c)
            
            sol = solve_ivp(lib.monodromy3,[0,self.tLC[-1]],init,
                            args=(self,),t_eval=self.tLC,
                            method=self.method,
                            rtol=1e-13,atol=1e-13)
            
            self.sol = sol.y.T
            self.M = np.reshape(self.sol[-1,:],(r,c))
            np.savetxt(self.monodromy_fname,self.M)

        else:
            self.M = np.loadtxt(self.monodromy_fname)
        
        if False:
            fig, axs = plt.subplots(nrows=self.dim,ncols=1,figsize=(10,10))
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
        
        print('eigenvalues',self.eigenvalues)
        print('eiogenvectors',self.eigenvectors)
        
        #print(self.eigenvectors)
        
        # print floquet multipliers
        
        
        einv = np.linalg.inv(self.eigenvectors/2)
        print('eig inverse',einv)
        
        idx = np.argsort(np.abs(self.eigenvalues-1))[0]
        #min_lam_idx2 = np.argsort(einv)[-2]
        
        
            
        self.g1_init = self.eigenvectors[:,self.min_lam_idx]/2.
        self.z0_init = einv[idx,:]
        self.i0_init = einv[self.min_lam_idx,:]

        print('min idx for prc',idx,)
        
        #print('Monodromy',self.M)
        #print('eigenvectors',self.eigenvectors)
        
        print('g1_init',self.g1_init)
        print('z0_init',self.z0_init)
        print('i0_init',self.i0_init)
        
        #print('Floquet Multiplier',self.lam)
        print('* Floquet Exponent kappa =',self.kappa)
        
        
        if False:
            fig, axs = plt.subplots(nrows=self.dim,
                                    ncols=self.dim,figsize=(10,10))
            
            for i in range(self.dim):
                for j in range(self.dim):
                    
                    axs[i,j].plot(self.tLC,self.sol[:,j+i*self.dim])
                
            axs[0,0].set_title('monodromy')
            plt.tight_layout()
            plt.show(block=True)
            time.sleep(.1)
        
        
    def load_g_sym(self):
        # load het. functions h if they exist. otherwise generate.
        #self.rule_g0 = {sym.Indexed('gx',0):s(0),sym.Indexed('gy',0):s(0)}
        
        # create dict of gv0=0,gh0=0,etc for substitution later.
        self.rule_g0 = {sym.Indexed('g'+name,0):
                        s(0) for name in self.var_names}
        
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
            
            # create symbolic derivative
            sym_collected = slib.generate_g_sym(self)  
            
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
                
                het_vec = self.interp_lam(i,self.g,fn_type='g')
                
                data = self.generate_g(i,het_vec)
                np.savetxt(self.g['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if True:
                fig, axs = plt.subplots(nrows=self.dim,ncols=1)
                
                for j,ax in enumerate(axs):
                    key = self.var_names[j]
                    ax.plot(self.tLC,data[:,j],label=key)
                    ax.legend()
                    
                axs[0].set_title('g'+str(i))
                print('g'+str(i)+' init',data[0,:])
                print('g'+str(i)+' final',data[-1,:])
                
                plt.tight_layout()
                plt.show(block=True)
                time.sleep(.1)
                
                
            self.g['dat'].append(data)
            
            for j,key in enumerate(self.var_names):
                #print(len(self.tLC),len(data[:,j]))
                fn_temp = interpb(self.tLC,data[:,j],self.T)
                imp_temp = imp_fn('g'+key+'_'+str(i),self.fmod(fn_temp))
                self.g['imp_'+key].append(imp_temp)
                
                lam_temp = lambdify(self.t,self.g['imp_'+key][i](self.t))
                self.g['lam_'+key].append(lam_temp)
                
            
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
    
    def generate_g(self,k,het_vec):
        """
        generate Floquet eigenfunctions g
        
        uses Newtons method
        """
        # load kth expansion of g for k >= 0
        
        if k == 0:
            # g0 is 0. dot his to keep indexing simple.
            return np.zeros((self.TN,len(self.var_names)))
        

        if k == 1:
            # pick correct normalization
            #init = [0,self.g1_init[1],self.g1_init[2],self.g1_init[3]]
            init = copy.deepcopy(self.g1_init)
        else:
            init = np.zeros(self.dim)
            
            # find intial condtion
        
        if k == 1:
            eps = 1e-2
            backwards = False
            rel_tol = 1e-7
            alpha = 1
        else:
            eps = 1e-2
            backwards = False
            rel_tol = 1e-9
            alpha = 1
            
            """
            if k == 3:
                backwards = True
                rel_tol = 1e-9
                alpha=0.2
                
            elif k == 4:
                backwards = True
                rel_tol = 1e-9
                alpha=0.7
                
            else:
                backwards = False
                rel_tol = 1e-7
                alpha=0.4
            """
                    
            init = lib.run_newton2(self,self.dg,init,k,het_vec,
                                  max_iter=20,eps=eps,
                                  rel_tol=rel_tol,rel_err=10,
                                  exception=False,alpha=alpha,
                                  backwards=backwards)
        
        # get full solution
        
        if backwards:
            tLC = -self.tLC
            
        else:
            tLC = self.tLC
            
        sol = solve_ivp(self.dg,[0,tLC[-1]],
                        init,args=(k,het_vec),
                        t_eval=tLC,method=self.method,
                        dense_output=True,
                        rtol=self.rtol,atol=self.atol)
        
        if backwards:
            gu = sol.y.T[::-1,:]
            
        else:
            gu = sol.y.T
        
            
        return gu


    def load_het_sym(self):
        # load het. for z and i if they exist. otherwise generate.
        
        for key in self.var_names:
            self.z['sym_'+key] = []
            self.i['sym_'+key] = []
        #    self.het1['sym_'+key] = []
        #self.het1 = {'sym_'+k: [] for k in self.var_names}
        
        # check that files exist
        val = 0
        for key in self.var_names:
            val += not(lib.files_exist(self.z['sym_fnames_'+key]))
            val += not(lib.files_exist(self.i['sym_fnames_'+key]))
        
        val += not(lib.files_exist([self.A_fname]))
        
        if val != 0:
            files_do_not_exist = True
        else:
            files_do_not_exist = False
        
        if self.recompute_het_sym or files_do_not_exist:
            
            sym_collected = self.generate_het_sym()
            
            for i in range(self.miter):
                for key in self.var_names:
                    
                    expr = sym_collected[key].coeff(self.psi,i)
                    expr = expr.subs(self.rule_g0)
                    self.z['sym_'+key].append(expr)
                    self.i['sym_'+key].append(expr)
                    #print('het1 key, i,expr', key, i,expr)
                    #print()
                    #print(self.g_sym_fnames[key][i])
                    dill.dump(self.z['sym_'+key][i],
                              open(self.z['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    dill.dump(self.i['sym_'+key][i],
                              open(self.i['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                
            # save matrix of a_i
            dill.dump(self.A,open(self.A_fname,'wb'),recurse=True)
            

        else:
            self.A, = lib.load_dill([self.A_fname])
            for key in self.var_names:
                self.z['sym_'+key] = lib.load_dill(self.z['sym_fnames_'+key])
                self.i['sym_'+key] = lib.load_dill(self.i['sym_fnames_'+key])
        
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
        self.a = {k: sym.zeros(self.dim,1) for k in self.var_names}
        
        #self.ax = Matrix([[0],[0]])
        #self.ay = Matrix([[0],[0]])
        
        for i in range(1,self.trunc_derivative+1):
            p1 = lib.kProd(i,self.dx_vec)
            p2 = kp(p1,sym.eye(self.dim))

            for j,key in enumerate(self.var_names):
                
                d1 = lib.vec(lib.df(self.rhs_sym[j],self.x_vec,i+1))
                #print((1/math.factorial(i)))
                self.a[key] += (1/math.factorial(i))*p2*d1
                
                
          
        self.A = sym.zeros(self.dim,self.dim)
        
        for i,key in enumerate(self.var_names):            
            self.A[:,i] = self.a[key]
        
        het = self.A*self.z['vec']
        
        # expand all terms
        out = {}
        for i,key in enumerate(self.var_names):
            het_key = sym.expand(het[i]).subs(self.rule_d2g)
            het_key = sym.collect(het_key,self.psi)
            het_key = sym.expand(het_key)
            het_key = sym.collect(het_key,self.psi)
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
                
                het_vec = self.interp_lam(i,self.z,fn_type='z')
                
                data = self.generate_z(i,het_vec)
                np.savetxt(self.z['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if True:
                fig, axs = plt.subplots(nrows=self.dim,ncols=1)
                
                for j,ax in enumerate(axs):
                    key = self.var_names[j]
                    ax.plot(self.tLC,data[:,j],label=key)
                    ax.legend()
                
                print('z'+str(i)+' init',data[0,:])
                print('z'+str(i)+' final',data[-1,:])
                axs[0].set_title('z'+str(i))
                plt.tight_layout()
                plt.show(block=True)
                time.sleep(.1)
                    
            self.z['dat'].append(data)
            
            for j,key in enumerate(self.var_names):
                
                fn_temp = interpb(self.tLC,data[:,j],self.T)
                imp_temp = imp_fn('z'+key+'_'+str(i),self.fmod(fn_temp))
                self.z['imp_'+key].append(imp_temp)
                
                lam_temp = lambdify(self.t,self.z['imp_'+key][i](self.t))
                self.z['lam_'+key].append(lam_temp)
            

        
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


        
        
    def generate_z(self,k,het_vec):
        
        if k == 0:
            init = copy.deepcopy(self.z0_init)
            #init = np.array([1.,1.,1.])
            eps = 1e-1
            #init = [-1.389, -1.077, 9.645, 0]
        else:
            
            if k == 1:
                exception = True
            else:
                exception = False
            
            init = np.zeros(self.dim)
            eps = 1e-1
            
            init = lib.run_newton2(self,self.dz,init,k,het_vec,
                                  max_iter=10,eps=eps,
                                  rel_tol=1e-8,
                                  exception=exception,
                                  backwards=False)
            
        sol = solve_ivp(self.dz,[0,-self.tLC[-1]],
                        init,args=(k,het_vec),
                        method=self.method,dense_output=True,
                        t_eval=-self.tLC,
                        rtol=self.rtol,atol=self.atol)
            
        zu = sol.y.T[::-1]
        #zu = sol.y.T
        
        if k == 0:
            # normalize
            v0,w0,q0 = [self.LC['lam_v'](0),
                        self.LC['lam_w'](0),
                        self.LC['lam_q'](0)]
            
            dLC = self.rhs(0,[v0,w0,q0])
            #zu = self.omega*zu/(np.dot(dLC,zu[0,:]))
            zu = zu/(np.dot(dLC,zu[0,:]))
            
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
                
                het_lams = self.interp_lam(i,self.i,fn_type='i')
                
                data = self.generate_i(i,het_lams)
                np.savetxt(self.i['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if True:
                fig, axs = plt.subplots(nrows=self.dim,ncols=1)
                
                for j,ax in enumerate(axs):
                    key = self.var_names[j]
                    ax.plot(self.tLC,data[:,j],label=key)
                    ax.legend()
                    
                print('i'+str(i)+' init',data[0,:])
                print('i'+str(i)+' final',data[-1,:])
                axs[0].set_title('i'+str(i))
                plt.tight_layout()
                plt.show(block=True)
                time.sleep(.1)
                
            self.i['dat'].append(data)
            
            for j,key in enumerate(self.var_names):
                fn_temp = interpb(self.tLC,data[:,j],self.T)
                imp_temp = imp_fn('i'+key+'_'+str(i),self.fmod(fn_temp))
                self.i['imp_'+key].append(imp_temp)
                
                lam_temp = lambdify(self.t,self.i['imp_'+key][i](self.t))
                self.i['lam_'+key].append(lam_temp)
                
            
               
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
        

        if k == 0:
            init = copy.deepcopy(self.i0_init)
            eps = 1e-2
            exception=False
            
        else:
            
            init = np.zeros(self.dim)
        
            if k == 1:
                exception = False
                eps = 1e-2
            else:
                exception = False
                eps = 1e-2
                
                
            init = lib.run_newton2(self,self.di,init,k,het_lams,
                                   max_iter=20,rel_tol=1e-9,
                                   rel_err=5,eps=eps,
                                   backwards=True,exception=exception)
        
        sol = solve_ivp(self.di,[0,-self.tLC[-1]],init,
                        args=(k,het_lams),
                        t_eval=-self.tLC,
                        method=self.method,dense_output=True,
                        rtol=self.rtol,atol=self.atol)
    
        iu = sol.y.T[::-1,:]
        
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
            
            F = self.rhs(0,[self.LC['lam_v'](0),
                            self.LC['lam_w'](0),
                            self.LC['lam_q'](0)])
            
            g1 = np.array([self.g['lam_v'][1](0),
                           self.g['lam_w'][1](0),
                           self.g['lam_q'][1](0)])
            
            z0 = np.array([self.z['lam_v'][0](0),
                           self.z['lam_w'][0](0),
                           self.z['lam_q'][0](0)])
            
            i0 = np.array([self.i['lam_v'][0](0),
                           self.i['lam_w'][0](0),
                           self.i['lam_q'][0](0)])
            
            J = self.jacLC(0)
            i1 = iu[0,:]
            
            ijg = np.dot(i0,np.dot(J,g1))
            be = (self.kappa - ijg - np.dot(i1,F))/(np.dot(z0,F))
            
            #print('actual',np.dot(F,i1))
            #print('expect',np.dot(i0,np.dot(self.kappa*self.eye-J,g1)))
            #print('canchg',z0)
            #print('amtchg',np.dot(F,z0))
            #print('mymult',be)
            #print('i1 unnormalized init',i1)
            
            
            
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
            
            self.cA['vec'] = sym.zeros(self.dim,1)
            self.cB['vec'] = sym.zeros(self.dim,1)
            
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
                
                collectedB = collect(expand(self.cB[key]),self.eps)
                collectedB = collect(expand(collectedB),
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
            
        
        
    def generate_k_sym(self):
        # generate terms involving the coupling term (see K in paper).
        
        # find K_i^{j,k}
        coupA = self.coupling(self.A_pair,option='sym')
        coupB = self.coupling(self.B_pair,option='sym')

        
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
        
        #self.ghx_sym_collected = sym.collect(self.hx[0],self.psi,evaluate=False)

    def interp_lam(self,k,fn_dict,fn_type='z'):
        """
        it is too slow to call individual interpolated functions
        in the symbolic heterogeneous terms.
        since the heterogeneous terms only depend on t, just make
        and interpolated version and use that instead so only 1 function
        is called for the het. terms per iteration in numerical iteration.
        """
        # lambdify heterogeneous terms for use in integration
        # create lambdified heterogeneous term and interpolate
        # load kth expansion of g for k >= 1
        
        # z and i use the same heterogeneous terms
        # so they are indistinguishable in this function.
        if fn_type == 'z' or fn_type == 'i':
            fn_type = 'z'
        
        rule = {}
        for key in self.var_names:
            tmp = {sym.Indexed(fn_type+key,i):fn_dict['imp_'+key][i](self.t)
                   for i in range(k)}
            #print(k,key,len(self.z['imp_'+key]))
            rule.update(tmp)
        
        rule = {**rule,**self.rule_LC,**self.rule_par}
        if fn_type == 'z':
            rule.update({**self.rule_g})
        
        het_imp = sym.zeros(1,self.dim)
        for i,key in enumerate(self.var_names):
            sym_fn = fn_dict['sym_'+key][k].subs(rule)
            lam = lambdify(self.t,sym_fn)
            
            # evaluate
            if fn_type == 'g' and (k == 0 or k == 1):
                y = np.zeros(self.TN)
            elif fn_type == 'z' and k == 0:
                y = np.zeros(self.TN)
            elif fn_type == 'i' and k == 0:
                y = np.zeros(self.TN)
            elif sym_fn == 0:
                y = np.zeros(self.TN)
            else:
                y = lam(self.tLC)
                
            # save as implemented fn
            interp = interpb(self.LC['t'],y,self.T)
            #print(key,sym_fn)
            #print('interp',key,sym_fn,interp(1))
            imp = imp_fn(key+'_het',self.fmod(interp))
            het_imp[i] = imp(self.t)
            
            
        het_vec = lambdify(self.t,het_imp)
        #print('het_vec',het_vec(1))
        
        if False and k > 0:
            fig, axs = plt.subplots(nrows=self.dim,ncols=1)
            for i,key in enumerate(self.var_names):
                print('k',k,key)                
                axs[i].plot(self.tLC*2,het_vec(self.tLC*2)[i])
            
            axs[0].set_title('lam dict')
            plt.tight_layout()
            plt.show(block=True)
            
        return het_vec
    
    
            

    def load_p_sym(self):
        """
        generate/load the het. terms for psi ODEs.
            
        to be solved using integrating factor meothod.
        
        pA['sym'][k] is the forcing function of order k
        """
        
        self.pA['sym'] = []
        #self.pB['sym'] = []
        
        print('* Computing... p symbolic')
        if self.recompute_p_sym or not(lib.files_exist(self.pA['sym_fnames'])):
            
            ircA = self.eps*self.i['vecA'].dot(self.cA['vec'])
            ircA = collect(expand(ircA),self.eps)
            ircA = collect(expand(ircA),self.eps)
            
            for i in range(self.miter):
                # save each order to list and dill.
                eps_i_termA = ircA.coeff(self.eps,i)
                self.pA['sym'].append(eps_i_termA)
                dill.dump(self.pA['sym'][i],
                          open(self.pA['sym_fnames'][i],'wb'),recurse=True)

        else:
            self.pA['sym'] = lib.load_dill(self.pA['sym_fnames'])
        
    def load_p(self):
        """
        generate/load the ODEs for psi.
        """
        
        #self.A_mg, self.B_mg = np.meshgrid(self.A_array,self.B_array)
        
        #self.interval,self.ds = np.linspace(0,300,10000,retstep=True)
        #self.ds = (self.interval[-1]-self.interval[0])/len(self.interval)
        #self.dxA = (self.A_array[-1]-self.A_array[0])/len(self.A_array)
        #self.dxB = (self.B_array[-1]-self.B_array[0])/len(self.B_array)
        
        # load all p or recompute or compute new.
        self.pA['dat'] = []
        self.pA['imp'] = []
        self.pA['lam'] = []
        
        #self.pA_data, self.pB_data, self.pA_imp, self.pB_imp = ([] for i in range(4))
        #self.pA_callable, self.pB_callable = ([] for i in range(2))

        # generate
        #if self.recompute_p or not(lib.files_exist(self.pA_fnames,self.pB_fnames)):
        
        print('* Computing...',end=' ')
        self.pool = _ProcessPool(processes=5)
        
        for i,fname in enumerate(self.pA['dat_fnames']):
            A_array,dxA = np.linspace(0,self.T,self.NA[i],retstep=True,
                                      endpoint=True)
            B_array,dxB = np.linspace(0,self.T,self.NB[i],retstep=True,
                                      endpoint=True)
            
            
            print('p_'+str(i),end=', ')
            if self.recompute_p or not(os.path.isfile(fname)):
                print('* Computing p'+str(i)+'...')
                start = time.time()
                pA_data = self.generate_p(i,A_array,dxA)
                end = time.time()
                print('time elapsed for p'+str(i)+' = '+str(end-start))
                
                np.savetxt(self.pA['dat_fnames'][i],pA_data)
                
            else:
                print('* Loading p'+str(i)+'...')
                pA_data = np.loadtxt(fname)
            
            if True:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                
                #ax.matshow(pA_data[190:,140:],cmap='viridis',aspect='auto')
                ax.matshow(pA_data,cmap='viridis',aspect='auto')
                #ax.set_xlim(0,self.NA[i])
                
                
                ax.set_ylabel('B[::-1]')
                ax.set_xlabel('A')
                ax.set_title('pA data'+str(i)\
                             +' NA='+str(self.NA[i])\
                             +' p_iter='+str(self.p_iter[i]))
                plt.show(block=True)
                plt.close()
            
            #print(np.shape(pA_data),np.shape(self.A_array),
            #      np.shape(self.B_array))
            # turn into interpolated 2d function (inputs automatically taken mod T)
            #print(np.shape(pA_data))
            pA_interp = interp2d(A_array,B_array,
                                 pA_data,bounds_error=False,
                                 fill_value=None)
            
            pA_interp2 = interp2db(pA_interp,self.T)
            
            pA_imp = imp_fn('pA_'+str(i),self.fLam2(pA_interp2))
            
            
            self.pA['dat'].append(pA_data)
            
            if i == 0:
                imp = imp_fn('pA_0', lambda x: 0)
                self.pA['lam'].append(imp)
                self.pA['imp'].append(imp)
            else:
                self.pA['imp'].append(pA_imp)
                self.pA['lam'].append(pA_interp)
                        
        self.pool.close()
        self.pool.join()
        self.pool.terminate()
        print()
        
        ta = self.thA
        tb = self.thB
        
        
        rule_pA = {sym.Indexed('pA',i):self.pA['imp'][i](ta,tb)
                       for i in range(self.miter)}
        rule_pB = {sym.Indexed('pB',i):self.pA['imp'][i](tb,ta)
                       for i in range(self.miter)}
        
        self.rule_p_AB = {**rule_pA,**rule_pB}
        
    def generate_p(self,k,A_array,dxA):
        import scipy as sp
        import numpy as np
        
        ta = self.thA
        tb = self.thB
        
        if k == 0:
            #pA0 is 0 (no forcing function)
            return np.zeros((self.NB[k],self.NA[k]))
        
        # put these implemented functions into the expansion
        ruleA = {sym.Indexed('pA',i):
                 self.pA['imp'][i](ta,tb) for i in range(k)}
        ruleB = {sym.Indexed('pB',i):
                 self.pA['imp'][i](tb,ta) for i in range(k)}
        
        
        rule = {**ruleA, **ruleB,
                **self.rule_g_AB,
                **self.rule_i_AB,
                **self.rule_LC_AB,
                **self.rule_par}
        
        ph_impA = self.pA['sym'][k].subs(rule)
        
        B_array = A_array
        
        # this lambidfy calls symbolic functions. slow.
        # convert lamdify to data and call linear interpolation on that.
        # then function call is same speed independent of order.
        lam_hetA = lambdify([ta,tb],ph_impA)
        lam_hetA_old = lam_hetA
        
        NA = self.NA[k]
        
        lam_hetA_data = np.zeros((NA,NA))
        
        for i in range(NA):
            ta2 = A_array[i]*np.ones_like(B_array)
            tb2 = B_array
            lam_hetA_data[:,i] = lam_hetA(ta2,tb2)
        
        het_interp = interp2d(A_array,B_array,
                              lam_hetA_data,bounds_error=False,
                              fill_value=None)
        
        #pA_imp = implemented_function('temp',self.myFunMod2A(het_interp))
        
        pA_interp2 = interp2db(het_interp,self.T)
        #lam_hetA = lambdify([ta,tb],pA_imp(ta,tb))
        
        A_mg, B_mg = np.meshgrid(A_array,B_array)
        
        # parallelize
        #s = copy.deepcopy(self.interval)
        kappa = self.kappa
       
        r,c = np.shape(A_mg)
        a_i = np.arange(self.NA[k],dtype=int)
        
        A_mg_idxs, B_mg_idxs = np.meshgrid(a_i,a_i)
        
        a_mg_idxs = np.reshape(A_mg_idxs,(r*c,))
        b_mg_idxs = np.reshape(B_mg_idxs,(r*c,))
        
        pA_data = np.zeros((r,c))
        pA_data = np.reshape(pA_data,(r*c,))
        
        idx = np.arange(r*c)
        exp = np.exp
        
        
        #i = 10
        T = self.T
        Ns = self.Ns[k]
        p_iter = self.p_iter[k]
        
        s,ds = np.linspace(0,p_iter*T,p_iter*Ns,retstep=True,endpoint=True)
        
        s = np.arange(0,T*p_iter,dxA)
        s_idxs = np.arange(len(s),dtype=int)
        exponential = exp(s*kappa)
        
        def return_integral(i):
            """
            return time integral at position a,b
            """
            
            a_idxs = np.mod(a_mg_idxs[i]-s_idxs,NA)
            b_idxs = np.mod(b_mg_idxs[i]-s_idxs,NA)
            
            periodic = lam_hetA_data[b_idxs,a_idxs]
            
            return np.sum(exponential*periodic)*dxA, i
        
        p = self.pool
        
        for x in tqdm.tqdm(p.imap(return_integral,idx,chunksize=1000),
                           total=r*c):
            integral, idx = x
            pA_data[idx] = integral
            
        pA_data = np.reshape(pA_data,(r,c))
        
        return pA_data
    
        
        
    def load_h_sym(self):
        # symbolic h terms
        
        self.hodd['sym'] = []
        
        if self.recompute_h_sym or not(lib.files_exist(self.hodd['sym_fnames'])):
            
            print('* Computing... H symbolic')
            #self.pA = Sum(eps**i_sym*Indexed('pA',i_sym),(i_sym,1,max_idx)).doit()
            z_rule = {}
            for key in self.var_names:
                for i in range(self.miter):
                    z_rule.update({Indexed('z'+key,i):Indexed('z'+key+'A',i)})

            #print('z_rule1',z_rule)
            #print()

            z_rule.update({self.psi:self.pA['expand']})
            
            #print('z_rule2',z_rule)
            #print()

            z = self.z['vec'].subs(z_rule)
            
            #print('ca vec',self.cA['vec'])
            #print('z',z)
            
            collected = collect(expand(self.cA['vec'].dot(z)),self.eps)
            
            self.h_collected = collect(expand(collected),self.eps)
            #print('hcollected',self.h_collected)
            
            for i in range(self.miter):
                collected = self.h_collected.coeff(self.eps,i)
                #print(collected)
                self.hodd['sym'].append(collected)
                dill.dump(self.hodd['sym'][i],
                          open(self.hodd['sym_fnames'][i],'wb'),
                          recurse=True)
                
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
        
        NA = self.NA[k]
        NB = self.NB[k]
        
        h = np.zeros(NB)
        
        A_array,dxA = np.linspace(0,self.T,NA,retstep=True,endpoint=True)
        B_array,dxB = np.linspace(0,self.T,NB,retstep=True,endpoint=True)
        
        t = A_array
        
        for j in range(NB):
            eta = B_array[j]
            h[j] = np.sum(self.hodd['lam'][k](t,t+eta))*dxA/self.T
        
        hodd = (h[::-1]-h)
        
        return hodd
            
        
    def bispeu(self,fn,x,y):
        """
        silly workaround
        https://stackoverflow.com/questions/47087109/...
        evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
        """
        return si.dfitpack.bispeu(fn.tck[0], fn.tck[1],
                                  fn.tck[2], fn.tck[3],
                                  fn.tck[4], x, y)[0][0]
        
    def dg(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous termsd
        
        order determines the Taylor expansion term
        """
        
        jac = self.jacLC(t)*(order > 0)
        
        #LC_vec = self.LC_vec(t)
        #jac = self.numerical_jac(self.rhs,LC_vec)*(order > 0)
        
        hom = np.dot(jac-order*self.kappa*self.eye,z)
        
        #het = np.array([het_lams['v'](t),
        #                het_lams['w'](t),
        #                het_lams['q'](t)])
    
        out = hom + het_vec(t)
    
        return out
    
    def dz(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        #LC_vec = self.LC_vec(t)
        #jac = self.numerical_jac(self.rhs,LC_vec).T
        
        #print(het_vec(t))
        
        jac = self.jacLC(t).T
        
        hom = np.dot(jac+order*self.kappa*self.eye,z)
        
        out = -hom - het_vec(t)
        
        return out
    
    def di(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = np.dot(self.jacLC(t).T+(order-1)*self.kappa*self.eye,z)
        
        out = -hom - het_vec(t)
        
        
        return out
    
    
    def fmod(self,fn):
        """
        input function-like. usually interp1d object
        
        needed to keep lambda input variable unique to fn.
        
        otherwise lambda will use the same input variable for 
        all lambda functions.
        """
        return lambda x=self.t: fn(np.mod(x,self.T))
    
    def fLam2(self,fn):
        """
        interp2db object
        """
        return lambda xA=self.thA,xB=self.thB: fn(xA,xB)
    
    def myFunMod2sym(self,fn):
        """
        same as above but for 2 variable function.
        
        fn: must be sympy object
        xA and xB must have same 1d array sizes. f(float,array) wont work.
        
        """
        
        return lambda xA=self.thA,xB=self.thB: fn(np.mod(xA,self.T),
                                                  np.mod(xB,self.T))
    
    
    def myFunMod2A(self,fn):
        """
        same as above but for 2 variable function for use with interp2d 
        function only.
        
        fn: must be interp2d function object
        xA and xB must have same 1d array sizes. f(float,array) wont work.
        
        """
        # need bispeu to allow for 1d array inputs.
        
        return lambda xA=self.thA,xB=self.thB: self.bispeu(fn,
                                                           np.mod(xA,self.T),
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
    
    a = MorrisLecar(recompute_LC=False,
                    recompute_monodromy=False,
                    recompute_g_sym=False,
                    recompute_g=False,
                    recompute_het_sym=False,
                    recompute_z=False,
                    recompute_i=False,
                    recompute_k_sym=False,
                    recompute_p_sym=False,
                    recompute_p=False,
                    recompute_h_sym=False,
                    recompute_h=False,
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
    
    for i in range(a.miter):
        lib.plot(a,'pA'+str(i))
    
    """
    for i in range(a.miter):
        lib.plot(a,'hodd'+str(i))
        
    
    
    
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
    __spec__  = None
    #import cProfile
    #import re
    #cProfile.runctx('main()',globals(),locals(),'profile.pstats')
    #cProfile.runctx('main()',globals(),locals())

    main()
