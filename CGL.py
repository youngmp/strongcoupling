# file for comparing to CGL. implement adjoint methods in Wilson 2020

# https://stackoverflow.com/questions/49306092/parsing-a-symbolic-expression-that-includes-user-defined-functions-in-sympy

# TODO: fix solutions scaled in amplitdue by 2pi. (5/27/2020)

"""
The logical flow of the class follows the paper by Wilson 2020.
-produce heterogeneous terms for g for arbirary dx
-substitute dx with g=g0 + psi*g1 + psi^2*g2+...
-produce het. terms for irc
-...

this file is also practice for creating a more general class for any RHS.

TODO:
    -make sure that np.dot and sym matrix products are consistent.
    -check that np.identity and sym.eye are consistent

"""


# user-defined
import MatchingLib as lib
from interp_basic import interp_basic as interpb
from interp2d_basic import interp2d_basic as interp2db
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

#from numba import njit
from scipy.fftpack import fft, ifft
from operator import methodcaller
from mpl_toolkits.mplot3d import Axes3D
from sympy import Matrix, symbols,diff, pi, Sum, Indexed, collect, expand
#from sympy import Function
from sympy import sympify as s
from sympy.physics.quantum import TensorProduct as kp
from sympy.utilities.lambdify import lambdify, implemented_function
from pathos.pools import _ProcessPool

imp_fn = implemented_function

#from interpolate import interp1d
#from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import interp2d
from scipy.integrate import solve_ivp, quad

matplotlib.rcParams.update({'figure.max_open_warning': 0})


class CGL(object):
    """
    Non-radial Isochron Clock
    Requires sympy, numpy, matplotlib.
    """
    
    def __init__(self,**kwargs):

        """
        recompute_gh : recompute het. terms for Floquet e.funs g
        """
        
        defaults = {
            'q_val':1,
            'eps_val':0,
            'd_val':1,
            
            'trunc_order':3,
            'trunc_derivative':2,
            
            'TN':10000,
            'dir':'dat',
            
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
            
        self.var_names = ['x','y']
        self.dim = len(self.var_names)
        
        # misc variables
        self.miter = self.trunc_order+1

        # Symbolic variables and functions
        self.eye = np.identity(self.dim)
        self.psi, self.eps, self.kap_sym = sym.symbols('psi eps kap_sym')
        
        self.rtol = 1e-8
        self.atol = 1e-8
        self.method = 'LSODA'
        self.rel_tol = 1e-6
        
        # for coupling computation. ctrl+F to see usage.
        self.NA = np.zeros(self.miter,dtype=int)+70
        self.NB = self.NA + 1
        self.Ns = np.zeros(self.miter,dtype=int)+50
        self.smax = np.zeros(self.miter,dtype=int)+1
        self.p_iter = np.zeros(self.miter,dtype=int)+5

        self.hNA = self.NA[0]
        self.hNB = self.NB[0]
        
        # parameters
        self.q, self.d = symbols('q d')
        
        self.rule_par = {self.q:self.q_val,self.d:self.d_val}
        
        # single-oscillator variables
        self.x, self.y, self.t, self.s = symbols('x y t s')
        self.dx, self.dy = symbols('dx dy')
        
        # coupling variables
        self.tA, self.tB, = symbols('tA tB')
        self.thA, self.psiA, self.thB, self.psiB = symbols('thA psiA thB psiB')
        
        self.xA, self.yA, self.xB, self.yB = symbols('xA yA xB yB')
        self.dxA, self.dyA, self.dxB, self.dyB = symbols('dxA dyA dxB dyB')
        
        self.vars = [self.x,self.y]
        self.A_vars = [self.xA,self.yA]
        self.dA_vars = [self.dxA,self.dyA]
        
        self.B_vars = [self.xB,self.yB]
        self.dB_vars = [self.dxB,self.dyB]
        
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
        
        
        self.dx_vec = Matrix([[self.dx,self.dy]])
        self.x_vec = Matrix([[self.x],[self.y]])
        
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
        self.dir = 'cgl_dat/'
        
        if (not os.path.exists(self.dir)):
            os.makedirs(self.dir)
        
        lib.generate_fnames(self)
        
        self.rhs_sym = self.rhs(0,self.vars,option='sym')
        
        # find limit cycle or load
        self.T = 2*np.pi/self.q_val
        self.omega = 2*np.pi/self.T
        
        #self.TN = TN
        # find limit cycle -- easy for this problem but think of general methods
        # see Dan's code for general method
        self.tLC = np.linspace(0,self.T,self.TN)
        self.LC['t'] = self.tLC
        
        # make interpolated version of LC
        #LC_interp_x = interp1d(tLC,LC_arr[:,0])
        #LC_interp_y = interp1d(tLC,LC_arr[:,1])
        
        self.LC['imp_x'] = sym.cos(self.q*self.t).subs(self.rule_par)
        self.LC['lam_x'] = lambdify(self.t,
                                    self.LC['imp_x'].subs(self.rule_par))
        
        self.LC['imp_y'] = sym.sin(self.q*self.t).subs(self.rule_par)
        self.LC['lam_y'] = lambdify(self.t,
                                    self.LC['imp_y'].subs(self.rule_par))
        
        print(self.LC['lam_y'](0.5))
        
        self.rule_LC = {}
        for i,key in enumerate(self.var_names):
            self.rule_LC.update({self.vars[i]:self.LC['imp_'+key]})
    
        
        slib.generate_expansions(self)
        slib.load_coupling_expansions(self)
        slib.load_jac_sym(self)
        
        # Run method
        # get monodromy matrix
        self.load_monodromy()
        
        if self.load_all:
            
            self.load_g_sym()  # get heterogeneous terms for g
            self.load_g()  # get g
            
            #t0 = time.time()
            self.load_het_sym()
            self.load_z()
            self.load_i()
            
            # LC A, B
            self.LC_xA = sym.cos(self.q_val*self.thA)
            self.LC_yA = sym.sin(self.q_val*self.thA)
    
            self.LC_xB = sym.cos(self.q_val*self.thB)
            self.LC_yB = sym.sin(self.q_val*self.thB)
            
            self.rule_LC_AB = {self.xA:self.LC_xA,
                               self.yA:self.LC_yA,
                               self.xB:self.LC_xB,
                               self.yB:self.LC_yB}
            
            # inputs to coupling function for oscillator A
            self.x_pairA = Matrix([[self.xA, self.yA, self.xB, self.yB]])
            self.dx_pairA = Matrix([[self.dxA, self.dyA, self.dxB, self.dyB]])
            
            # inputs to coupling function for oscillator B
            self.x_pairB = Matrix([[self.xB, self.yB, self.xA, self.yA]])
            self.dx_pairB = Matrix([[self.dxB, self.dyB, self.dxA, self.dyA]])
            
            #self.generate_coupling_expansions()
            self.load_k_sym()
            
            self.load_p_sym()
            self.load_p()
            
            self.load_h_sym()
            self.load_h()
        
    def rhs(self,t,z,option='value'):
        """
        right-hand side of the equation of interest. CCGL model.
        
        write in standard python notation as if it will be used in an ODE solver.

        Returns
        -------
        right-hand side equauation in terms of the inputs. if x,y scalars, return scalar.
        if x,y, sympy symbols, return symbol.

        """
        
        x,y = z
        
        R2 = x**2 + y**2
        
        if option == 'value':
            return np.array([x*(1-R2)-self.q_val*R2*y,
                             y*(1-R2)+self.q_val*R2*x])
        elif option == 'sym':
            return Matrix([x*(1-R2)-self.q*R2*y,
                           y*(1-R2)+self.q*R2*x])
    
    
    def coupling(self,vars_pair,option='value'):
        """
        r^(2n) to r^n function. default parameter order is from perspective of
        first oscillator.
        
        in this case the input is (x1,y1,x2,y2) and the output is an r2 vec.
        """
        x1,y1,x2,y2 = vars_pair
        
        if option == 'value':
            return np.array([x2-x1-self.d_val*(y2-y1),y2-y1+self.d_val*(x2-x1)])
        elif option == 'sym':
            return Matrix([x2-x1-self.d*(y2-y1),y2-y1+self.d*(x2-x1)])
    
    def load_monodromy(self):
        """
        if monodromy data exists, load. if DNE or recompute required, compute here.
        """
        if self.recompute_monodromy\
            or not(lib.files_exist([self.monodromy_fname])):
            init = np.reshape(self.eye,self.dim**2)
            
            sol = solve_ivp(lib.monodromy,[0,self.tLC[-1]],init,
                            args=(self.jacLC,),
                            method=self.method,dense_output=True,
                            rtol=self.rtol,atol=self.atol)
            
            self.sol = sol.sol(self.tLC).T
            self.M = np.reshape(self.sol[-1,:],(self.dim,self.dim))
            np.savetxt(self.monodromy_fname,self.M)

        else:
            self.M = np.loadtxt(self.monodromy_fname)
        
        
        
        #print('Monodromy Matrix',self.M)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.M)
        #print(self.eigenvalues)
        
        # get smallest eigenvalue and associated eigenvector
        self.min_lam_idx = np.argmin(self.eigenvalues)
        self.lam = self.eigenvalues[self.min_lam_idx]  # floquet mult.
        self.kappa = np.log(self.lam)/self.T  # floquet exponent
        
        self.g1_init = self.eigenvectors[:,self.min_lam_idx]
        self.z0_init = np.linalg.inv(self.eigenvectors)[1,:]
        self.i0_init = np.linalg.inv(self.eigenvectors)[0,:]  # figure out how to pick inits.
        
        #print('g1 init',self.g1_init)
        #print('z0 init',self.z0_init)
        #print('i0 init',self.i0_init)
        
        #print(self.z0_init,self.i0_init)
        
        #print('Floquet Multiplier',self.lam)
        print('Floquet Exponent kapa =',self.kappa)
        
    def load_g_sym(self):
        # load het. functions h if they exist. otherwise generate.
        self.rule_g0 = {sym.Indexed('g'+name,0):
                        s(0) for name in self.var_names}
        
        #self.rule_g0 = {sym.Indexed('gx',0):s(0),sym.Indexed('gy',0):s(0)}
        for key in self.var_names:
            self.g['sym_'+key] = []
        
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
            
    def generate_g_sym(self):
        """
        generate heterogeneous terms for the Floquet eigenfunctions g.
        
        purely symbolic.

        Returns
        -------
        list of symbolic heterogeneous terms in self.ghx_sym, self.ghy_sym.

        """
        
        # get the general expression for h before plugging in g.
        self.hx = 0
        self.hy = 0
        
        for i in range(2,self.miter):
            # all x1,x2 are evaluated on limit cycle x=cos(t), y=sin(t)
            p = lib.kProd(i,self.dx_vec)
            d1 = lib.vec(lib.df(self.CGL_sym[0],self.x_vec,i))
            d2 = lib.vec(lib.df(self.CGL_sym[1],self.x_vec,i))
            
            
            self.hx += (1/math.factorial(i)) * np.dot(p,d1)
            self.hy += (1/math.factorial(i)) * np.dot(p,d2)
            
        self.hx = sym.Matrix(self.hx)
        self.hy = sym.Matrix(self.hy)
        
        # expand all terms
        self.hx = sym.expand(self.hx.subs([(self.dx,self.gx),(self.dy,self.gy)]))
        self.hy = sym.expand(self.hy.subs([(self.dx,self.gx),(self.dy,self.gy)]))
        
        # collect all psi terms into dict with psi**k as index.
        self.ghx_sym_collected = sym.collect(self.hx[0],self.psi,evaluate=False)
        self.ghy_sym_collected = sym.collect(self.hy[0],self.psi,evaluate=False)
        
    
    def load_g(self):
        
        # load all g or recompute or compute new.
        
        self.g['dat'] = []
        
        for key in self.var_names:
            self.g['imp_'+key] = []
            self.g['lam_'+key] = []
        
        print('* Computing...', end=' ')
        for i in range(self.miter):
            print('g_'+str(i),end=', ')
            fname = self.g['dat_fnames'][i]
            #print('i,fname',i,fname)
            file_does_not_exist = not(os.path.isfile(fname))

            if self.recompute_g or file_does_not_exist:
                #print('g no exist',fname)
                het_vec = self.interp_lam(i,self.g,fn_type='g')                
                data = self.generate_g(i,het_vec)
                
                np.savetxt(self.g['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if False:
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
                #print('fn_temp',len(self.tLC),len(data[:,j]))
                #print(fname)
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
            backwards = True
            rel_tol = 1e-7
            alpha = 1
        else:
            eps = 1e-2
            backwards = True
            rel_tol = 1e-8
            alpha = 1
            
            init = lib.run_newton2(self,self.dg,init,k,het_vec,
                                  max_iter=100,eps=eps,
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

        print('computed g?')
        if backwards:
            gu = sol.y.T[::-1,:]
            
        else:
            gu = sol.y.T
        
            
        return gu
        
    def load_sols(self,fnames,symName='g',varnum=1):
        """
        Parameters
        ----------
        fnames : list
            list of file names. single file names should be intered as [fname]
        name : str, optional
            load solutions g, z, or i. The default is 'g'.
        varnum : number of independent variables
            
        Returns
        -------
        list1 : TYPE
            DESCRIPTION.
        listx : TYPE
            DESCRIPTION.
        listy : TYPE
            DESCRIPTION.

        """
        
        list1, listx, listy = ([] for i in range(3))
        
        for i in range(len(fnames)):
            data = np.loadtxt(fnames[i])
            
            list1.append(data)
            
            if varnum == 1:
                fnx = interpb(self.tLC,data[:,0],self.T)
                fny = interpb(self.tLC,data[:,1],self.T)
                
                xtemp = implemented_function(symName+'x_'+str(i), self.myFunMod(fnx))
                ytemp = implemented_function(symName+'y_'+str(i), self.myFunMod(fny))
                
            else:
                #print('loaded pa', np.shape(data))
                #print(np.shape(self.A_array),np.shape(self.B_array))
                if symName == 'pA':
                    fnx = interp2d(self.A_array,self.B_array,data,bounds_error=False,
                                   fill_value=None)
                if symName == 'pB':
                    fnx = interp2d(self.B_array,self.A_array,data,bounds_error=False,
                                   fill_value=None)
                    
                xtemp = implemented_function(symName+'_'+str(i),self.myFunMod2A(fnx))
                ytemp = fnx
                
                
            listx.append(xtemp)
            listy.append(ytemp)
            
        return list1, listx, listy

    
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
        
        for i in range(1,self.miter):
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
            file_does_not_exist = not(os.path.isfile(fname))
            #print('z fname',fname)
            if self.recompute_z or file_does_not_exist:
                
                het_vec = self.interp_lam(i,self.z,fn_type='z')
                
                data = self.generate_z(i,het_vec)
                np.savetxt(self.z['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if False:
                fig, axs = plt.subplots(nrows=self.dim,ncols=1)
                
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
            init = np.zeros(self.dim)
            eps = 1e-1
            
            init = lib.run_newton2(self,self.dz,init,k,het_vec,
                                  max_iter=100,eps=eps,
                                  rel_tol=1e-8,rel_err=10,
                                  backwards=True)
            
        sol = solve_ivp(self.dz,[0,-self.tLC[-1]],
                        init,args=(k,het_vec),
                        method=self.method,dense_output=True,
                        t_eval=-self.tLC,
                        rtol=self.rtol,atol=self.atol)
            
        zu = sol.y.T[::-1]
        #zu = sol.y.T
        
        if k == 0:
            # normalize
            x0,y0 = [self.LC['lam_x'](0),
                        self.LC['lam_y'](0)]
            
            dLC = self.rhs(0,[x0,y0])
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
            file_does_not_exist = not(os.path.isfile(fname))
            
            if self.recompute_i or file_does_not_exist:
                
                het_lams = self.interp_lam(i,self.i)
                
                data = self.generate_i(i,het_lams)
                np.savetxt(self.i['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if False:
                fig, axs = plt.subplots(nrows=self.dim,ncols=1)
                
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
            #print(init)
            #sol = solve_ivp(self.di,[0,-self.tLC[-1]],init,
            #                args=(k,),
            #                t_eval=-self.tLC,
            #                method=self.method,dense_output=True,
            #                rtol=self.rtol,atol=self.atol)
            
            #iu = sol.y.T[::-1,:]
            
        else:
            
            #print('het i',het_lams['v'](1))
            init = np.zeros(self.dim)
        
            if k == 1:
                exception = False
                eps = 1e-2
            else:
                exception = False
                eps = 1e-2
                
                
            init = lib.run_newton2(self,self.di,init,k,het_lams,
                                   max_iter=100,rel_tol=1e-8,
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
            
            F = self.rhs(0,[self.LC['lam_x'](0),
                            self.LC['lam_y'](0)])
            
            g1 = np.array([self.g['lam_x'][1](0),
                           self.g['lam_y'][1](0)])
            
            z0 = np.array([self.z['lam_x'][0](0), 
                           self.z['lam_y'][0](0)])
            
            i0 = np.array([self.i['lam_x'][0](0),
                           self.i['lam_y'][0](0)])
            
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
        self.pool = _ProcessPool(processes=10)
        
        for i,fname in enumerate(self.pA['dat_fnames']):
            A_array,dxA = np.linspace(0,self.T,self.NA[i],retstep=True,
                                      endpoint=True)
            B_array,dxB = np.linspace(0,self.T,self.NB[i],retstep=True,
                                      endpoint=True)
            
            
            print('p_'+str(i),end=', ')
            if self.recompute_p or not(os.path.isfile(fname)):
                
                pA_data = self.generate_p(i,A_array,B_array)
                np.savetxt(self.pA['dat_fnames'][i],pA_data)
                
            else:
                pA_data = np.loadtxt(fname)
            
            pA_interp = interp2d(A_array,B_array,
                                 pA_data,bounds_error=False,
                                 fill_value=None)
            
            pA_interp2 = interp2db(pA_interp,self.T)
            
            pA_imp = implemented_function('pA_'+str(i),self.fLam2(pA_interp2))
            
            
            self.pA['dat'].append(pA_data)
            
            if i == 0:
                self.pA['imp'].append(implemented_function('pA_0', lambda x: 0))
                self.pA['lam'].append(0)
            
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
        
    def generate_p(self,k,A_array,B_array):
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
        
        #print('k, rule:',k,rule)
        #print('k, self.pA[sym][k]:',k,self.pA['sym'][k])
        ph_impA = self.pA['sym'][k].subs(rule)
        
        # this lambidfy calls symbolic functions. slow.
        # convert lamdify to data and call linear interpolation on that.
        # then function call is same speed independent of order.
        lam_hetA = lambdify([ta,tb],ph_impA)
        lam_hetA_old = lam_hetA
        
        
        # maximize accuracy of interpolation
        NA2 = self.TN
        NB2 = NA2+1
        lam_hetA_data = np.zeros((NB2,NA2))
        
        A_array2 = np.linspace(0,self.T,NA2,endpoint=True)
        B_array2 = np.linspace(0,self.T,NB2,endpoint=True)
        
        A_mg2, B_mg2 = np.meshgrid(A_array2,B_array2)
        
        
        
        """
        for i in range(len(A_array2)):
            print('i temp',i)
            for j in range(len(B_array2)):
                ta2 = A_array2[i]
                tb2 = B_array2[j]
                lam_hetA_data[j,i] = lam_hetA(ta2,tb2)
            
            aa = lam_hetA(A_array2[i]*np.ones_like(B_array2[:7]),B_array2[:7])
            print(lam_hetA_data[:7,i])
            print(aa[:7])
        """
        
        
        for i in range(len(A_array2)):
            
            ta2 = A_array2[i]*np.ones_like(B_array2)
            tb2 = B_array2
            lam_hetA_data[:,i] = lam_hetA(ta2,tb2)
            
        
        het_interp = interp2d(A_array2,B_array2,
                              lam_hetA_data,bounds_error=False,
                              fill_value=None)
        
        het_interp2 = interp2db(het_interp,self.T)
        
        #pA_imp = implemented_function('temp',self.fLam2(het_interp2))
        #pA_interp2 = lambdify([ta,tb],pA_imp(ta,tb))
        
        
        A_mg, B_mg = np.meshgrid(A_array,B_array)
        
        # parallelize
        #s = copy.deepcopy(self.interval)
        kappa = self.kappa
       
        r,c = np.shape(A_mg)
        a = np.reshape(A_mg,(r*c,))
        b = np.reshape(B_mg,(r*c,))
        
        pA_data = np.zeros((r,c))
        pA_data = np.reshape(pA_data,(r*c,))
        
        
        #i = 10
        T = self.T
        smax = self.smax[k]
        Ns = self.Ns[k]
        p_iter = self.p_iter[k]
        
        s1,ds1 = np.linspace(0,T,Ns,retstep=True,endpoint=True)
        s,ds = np.linspace(0,p_iter*T,p_iter*Ns,retstep=True,endpoint=True)
        
        idx = np.arange(len(a))
        exp = np.exp
        
        
        def return_integral(i):
            #return time integral at position a,b
            
            #val = np.sum(pA_interp2(a[i]-s,b[i]-s)*exp(kappa*s))*ds
            #return val,i
            
            pA_one_period = het_interp2(a[i]-s1,b[i]-s1)
            val = 0
            for j in range(p_iter):
                s_interval = np.linspace(j*T,(j+1)*T,Ns,endpoint=True)
                val += np.sum(pA_one_period*exp(kappa*s_interval))*ds1
            return val,i
        
        """
        self.interval = np.linspace(0,4,50)
        self.ds = (self.interval[-1]-self.interval[0])/len(self.interval)
        s = self.interval
        for i in range(r*c):
            
            
            intA = np.exp(self.kappa*s)*lam_hetA(a[i]-s,b[i]-s)
            
            #intA = np.exp(self.kappa*s)*pA_interp2(a[i]-s,b[i]-s)
            
            
            #err = np.amax(np.abs(intAa-intA))
            
            if False and (i % 100 == 0) and (err > 1e-4):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(intAa)
                ax.plot(intA)
                plt.show(block=True)
                print(i,err)
                time.sleep(.1)
            
            pA_data[i] = np.sum(intA)*self.ds
            
            #integral, idx = return_integral(i)
            #pA_data[i] = integral
            
        pA_data = np.reshape(pA_data,(r,c))
        
        """
        
        
        p = self.pool
        
        print()
        for x in tqdm.tqdm(p.imap(return_integral,idx,chunksize=200),
                           total=len(a)):
            integral, idx = x
            pA_data[idx] = integral
            
            #sys.stderr.write('\rdone {:.0%}'.format(i/len(s)))
        
        
        pA_data = np.reshape(pA_data,(r,c))
        
        
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.matshow(pA_data,cmap='viridis')
            
            ax.set_ylabel('A')
            ax.set_xlabel('B')
            ax.set_title('pA data'+str(k))
            plt.show(block=True)
            plt.close()
        
        return pA_data
        """
    
        pA_data = np.zeros((self.NB[k],self.NA[k]))
        
        # no choice but to double loop because of interp2d.
        for i in range(self.NA[k]):
                
            for j in range(self.NB[k]):
                a, b = A_array[i], B_array[j]
                
                intA = np.exp(self.kappa*s)*lam_hetA(a-s,b-s)
                #intA = np.exp(self.kappa*s)*pA_interp2(a-s,b-s)
                
                pA_data[j,i] = np.sum(intA)*ds
        
        
        
        
        return pA_data
        """
        
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

            z_rule.update({self.psi:self.pA['expand']})
            z = self.z['vec'].subs(z_rule)
            
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
        
        h_mg = np.zeros((self.NB[k],self.NA[k]))
        
        
        A_array,dxA = np.linspace(0,self.T,self.NA[k],retstep=True,
                                  endpoint=True)
        B_array,dxB = np.linspace(0,self.T,self.NB[k],retstep=True,
                                  endpoint=True)
        
        for j in range(self.NB[k]):
            t = A_array
            eta = B_array[j]
            
            h_mg[j,:] = self.hodd['lam'][k](t,t+eta)
        
        # for i in range(self.NA[k]):
        #     for j in range(self.NB):
        #         t = self.A_array[i]
        #         eta = self.B_array[j]
                
        #         h_mg[j,i] = self.h_lams[k](t,t+eta)
                
        # sum along axis to get final form
        h = np.sum(h_mg,axis=1)*dxA/self.T
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(h)
            title = ('h non-odd '+str(k)
                     +'NA='+str(self.NA[k])
                     +', NB='+str(self.NB[k])
                     +', Ns='+str(self.Ns[k]))
            ax.set_title(title)
            plt.show(block=True)
        
        #print(h)
        #hodd = h
        hodd = (h[::-1]-h)
        
        return hodd
        
    def bispeu(self,fn,x,y):
        """
        silly workaround
        https://stackoverflow.com/questions/47087109/evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
        """
        return si.dfitpack.bispeu(fn.tck[0], fn.tck[1], fn.tck[2], fn.tck[3], fn.tck[4], x, y)[0][0]
        
    def dg(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        
        jac = self.jacLC(t)*(order > 0)
        
        hom = np.dot(jac-order*self.kappa*self.eye,z)
        #het = 0.5*np.array([hetx(t),hety(t)])
        
        #if order == 1:
        #    print(t,jac,hom,het,self.jacLC(self.t))
        
        return hom + het_vec(t)
    
    def dz(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = -np.dot(self.jacLC(t).T+order*self.kappa*self.eye,z)
        #het = -np.array([hetx(t),hety(t)])
        
        out = hom - het_vec(t)
        
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
        
        
        
        hom = -np.dot(self.jacLC(t).T+self.kappa*(order-1)*self.eye,z)
        #het = -np.array([hetx(t),hety(t)])
        
        out = hom - het_vec(t)
        
        #if order == 0 and int(t*self.TN) % 5 == 0:
        #    print(t,z,out,int(t*self.TN))
        
        return out
    
    
    
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
        #print('self lc',self.LC)
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
        
        #het_lams = {}
        het_imp = sym.zeros(1,self.dim)
        for i,key in enumerate(self.var_names):
            sym_fn = fn_dict['sym_'+key][k].subs(rule)
            lam = lambdify(self.t,sym_fn)
            #print('lam',lam(10))
            #lam = lambdify(self.t,fn_dict['sym_'+key][k].subs(rule))
            
            
            # evaluate
            if fn_type == 'g' and (k == 0 or k == 1):
                y = np.zeros(self.TN)
            elif fn_type == 'z' and k == 0:
                y = np.zeros(self.TN)
            elif fn_type == 'i' and k == 0:
                y = np.zeros(self.TN)
            else:
                #print('lam',k,len(self.tLC),lam(1))
                y = lam(self.tLC)
                
                
            # save as implemented fn
            interp = interpb(self.LC['t'],y,self.T)
            imp = imp_fn(key,self.fmod(interp))
            het_imp[i] = imp(self.t)
            
            
        het_vec = lambdify(self.t,het_imp)
        
        #print('print het vec',het_vec(1))
        
        if False and k > 0:
            fig, axs = plt.subplots(nrows=self.dim,ncols=1)
            for i,key in enumerate(self.var_names):
                print('k',k,key)                
                axs[i].plot(self.tLC*2,het_vec(self.tLC*2)[i])
            
            axs[0].set_title('lam dict')
            plt.tight_layout()
            plt.show(block=True)
            
        return het_vec
    
    
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
    
    
    def myFunMod(self,fn):
        """
        input function-like. usually interp1d object
        
        needed to keep lambda input variable unique to fn.
        
        otherwise lambda will use the same input variable for all lambda functions.
        """
        return lambda x=self.t: fn(np.mod(x,self.T))
    
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
        same as above but for 2 variable function for use with interp2d function only.
        
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
        return lambda x=self.s,xA=self.tA,xB=self.tB: fn(np.mod(xA-x,self.T),np.mod(xB-x,self.T))



def main():
    
    # for NIC, 3rd derivatives go away, so we only need trunc_gh=3.
    a = CGL(recompute_monodromy=False,
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
            trunc_order=5,
            trunc_derviative=5,
            d_val=1,
            q_val=1,
            TN=2001,
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
    import cProfile
    import re
    cProfile.runctx('main()',globals(),locals(),'profile.pstats')
    #cProfile.runctx('main()',globals(),locals())

    #main()
