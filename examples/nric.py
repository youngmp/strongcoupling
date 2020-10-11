# https://stackoverflow.com/questions/49306092/parsing-a-symbolic-expression-that-includes-user-defined-functions-in-sympy

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

import inspect
import time
import os
import math
#import time
import dill
import copy

import numpy as np
#import scipy as sp
import sympy as sym
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sympy import Matrix, symbols,diff, pi
#from sympy import Function
from sympy import sympify as s
from sympy.physics.quantum import TensorProduct as kp
from sympy.utilities.lambdify import lambdify, implemented_function
#from sympy.interactive import init_printing

from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp

matplotlib.rcParams.update({'figure.max_open_warning': 0})


class NIC(object):
    """
    Non-radial Isochron Clock
    Requires sympy, numpy, matplotlib.
    """
    
    def __init__(self,
                 SIG=0.08,
                 RHO=0.12,
                 P=2*np.pi,
                 trunc_g=6,
                 trunc_gh=3,
                 TN=2000,
                 dir='dat',
                 
                 recompute_monodromy=True,
                 recompute_gh=False,
                 recompute_g=False,
                 
                 recompute_het=False,
                 recompute_z=False,
                 recompute_i=False,
                 
                 ):

        """
        recompute_gh : recompute het. terms for Floquet e.funs g
        
        
        """

        # Model parameters
        self.SIG = SIG
        self.RHO = RHO
        self.P = P
        
        # Simulation/method parameters
        self.trunc_g = trunc_g  # max power term in psi of Floquet e.fun g
        self.trunc_gh = trunc_gh  # max in summation for heterogeneous term
        
        self.recompute_monodromy = recompute_monodromy
        self.recompute_gh = recompute_gh
        self.recompute_g = recompute_g
        
        self.recompute_het = recompute_het
        self.recompute_z = recompute_z
        self.recompute_i = recompute_i
        
        # find limit cycle or load
        self.T = 1
        self.omega = 2*np.pi
        self.TN = TN
        # find limit cycle -- easy for this problem but think of general methods
        # see Dan's code for general method
        self.tLC = np.linspace(0,self.T,self.TN)
        #LC_arr = odeint(rhs,[1,0],tLC,args=(f,))
        
        # make interpolated version of LC
        #LC_interp_x = interp1d(tLC,LC_arr[:,0])
        #LC_interp_y = interp1d(tLC,LC_arr[:,1])
        
        
        # filenames and directories
        self.dir = 'dat/'
        
        if (not os.path.exists(self.dir)):
            os.makedirs(self.dir)
            
        self.model_params = '_sig=' + str(self.SIG) \
                            + '_rho=' + str(self.RHO) \
                            + '_P=' + str(self.P)
        self.sim_params = '_TN='+str(self.TN)
        
        self.monodromy_fname = self.dir+'monodromy_'+self.model_params+self.sim_params+'.txt'

        self.ghx_fnames = [self.dir+'ghx_'+str(i)+self.model_params+self.sim_params+'.d' 
                           for i in range(self.trunc_g+1)]
        self.ghy_fnames = [self.dir+'ghy_'+str(i)+self.model_params+self.sim_params+'.d' 
                           for i in range(self.trunc_g+1)]
        self.g_fnames = [self.dir+'g_'+str(i)+self.model_params+self.sim_params+'.txt' 
                         for i in range(self.trunc_g+1)]
        
        self.hetx_fnames = [self.dir+'hetx_'+str(i)+self.model_params+self.sim_params+'.d' 
                            for i in range(self.trunc_g+1)]
        self.hety_fnames = [self.dir+'hety_'+str(i)+self.model_params+self.sim_params+'.d' 
                            for i in range(self.trunc_g+1)]
        self.A_fname = self.dir+'A_'+self.model_params+self.sim_params+'.d'
        
        self.z_fnames = [self.dir+'z_'+str(i)+self.model_params+self.sim_params+'.txt' 
                         for i in range(self.trunc_g+1)]
        self.i_fnames = [self.dir+'i_'+str(i)+self.model_params+self.sim_params+'.txt' 
                         for i in range(self.trunc_g+1)]
    
        # Symbolic variables and functions
        self.eye = np.identity(2)
        
        self.psi = sym.symbols('psi')
        self.x1,self.x2,self.x3,self.t = symbols('x1 x2,x3, t')
        self.dx1,self.dx2,self.dx3 = symbols('dx1 dx2 dx3')
        
        self.dx = Matrix([[self.dx1,self.dx2]])
        self.x = Matrix([[self.x1,self.x2]])
    
        R2 = self.x1**2+self.x2**2
        self.NIC1 = self.P*(self.SIG*self.x1*(1-R2) - self.x2*(1+self.RHO*(R2-1)))
        self.NIC2 = self.P*(self.SIG*self.x2*(1-R2) + self.x1*(1+self.RHO*(R2-1)))
        #self.rhs_sym = Matrix([self.NIC1,self.NIC2])
        
        # symbol J on LC.
        self.jacLC_symbolic = Matrix([[diff(self.NIC1,self.x1),diff(self.NIC1,self.x2)],
                                      [diff(self.NIC2,self.x1),diff(self.NIC2,self.x2)]]).subs(
                                      [(self.x1,sym.cos(2*pi*self.t)),
                                       (self.x2,sym.sin(2*pi*self.t))])
        
        # make RHS and Jacobian callable functions
        self.f = lambdify((self.x1,self.x2),[self.NIC1,self.NIC2])
        
        #jac = sym.lambdify((x1,x2),Jac)
        self.jacLC = lambdify((self.t),self.jacLC_symbolic)
        
        
        # assume gx is the first coordinate of Floquet eigenfunction g
        # brackets denote Taylor expansion functions
        # now substitute Taylor expansion dx = gx[0] + gx[1] + gx[2] + ...
        i_sym = sym.symbols('i_sym')  # summation index
        self.gx = sym.Sum(self.psi**i_sym*sym.Indexed('gx',i_sym),(i_sym,0,self.trunc_g)).doit()
        self.gy = sym.Sum(self.psi**i_sym*sym.Indexed('gy',i_sym),(i_sym,0,self.trunc_g)).doit()
        
        self.zx = sym.Sum(self.psi**i_sym*sym.Indexed('zx',i_sym),(i_sym,0,self.trunc_g)).doit()
        self.zy = sym.Sum(self.psi**i_sym*sym.Indexed('zy',i_sym),(i_sym,0,self.trunc_g)).doit()
        
        self.ix = sym.Sum(self.psi**i_sym*sym.Indexed('ix',i_sym),(i_sym,0,self.trunc_g)).doit()
        self.iy = sym.Sum(self.psi**i_sym*sym.Indexed('iy',i_sym),(i_sym,0,self.trunc_g)).doit()
        
        # Run method
        self.load_monodromy()  # get monodromy matrix
        self.load_gh()  # get heterogeneous terms for g
        self.load_g()  # get g
        
        #t0 = time.time()
        self.load_het()
        self.load_z()
        self.load_i()
        
        # create lambdified versions
        
        
        #t1 = time.time()
        #print('*\t Run time PRC + IRC',t1-t0)
        
        #self.load_ih()
        
        #print('*\t Run time IRC',t1-t0)
        
    def load_monodromy(self):
        """
        if monodromy data exists, load. if DNE or recompute required, compute here.
        """
        if self.recompute_monodromy or not(lib.files_exist([self.monodromy_fname])):
            self.generate_monodromy()
            np.savetxt(self.monodromy_fname,self.M)

        else:
            self.M = np.loadtxt(self.monodromy_fname)
        
        print('Monodromy Matrix',self.M)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.M)
        
        # get smallest eigenvalue and associated eigenvector
        self.min_idx = np.argmin(self.eigenvalues)
        self.lam = self.eigenvalues[self.min_idx]  # floquet mult.
        self.kappa = np.log(self.lam)  # floquet exponent
        
        self.g1_init = self.eigenvectors[:,self.min_idx]
        self.z0_init = np.linalg.inv(self.eigenvectors)[1,:]
        self.i0_init = np.linalg.inv(self.eigenvectors)[0,:]  # figure out how to pick inits.
        
        #print(self.z0_init,self.i0_init)
        
        print('Floquet Multiplier',self.lam)
        print('Floquet Exponent',np.log(self.lam))
        print('Eigenvectors of M',self.eigenvectors[:,0],self.eigenvectors[:,1])
        
    def generate_monodromy(self):
        # Get Monodromy matrix
        #self.sol = odeint(lib.monodromy,np.reshape(self.eye,4),self.tLC,args=(self.jacLC,))
        
        init = np.reshape(self.eye,4)
        sol = solve_ivp(lib.monodromy,[0,self.tLC[-1]],init,
                        args=(self.jacLC,),
                        method='RK45',dense_output=True)
            
        self.sol = sol.sol(self.tLC).T
    
        if True:
        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.tLC,self.sol[:,0])
            ax.plot(self.tLC,self.sol[:,1])
            ax.plot(self.tLC,self.sol[:,2])
            ax.plot(self.tLC,self.sol[:,3])
            ax.set_title('Monodromy sol')
            plt.show(block=True)
        
        self.M = np.reshape(self.sol[-1,:],(2,2))
        
        
    def load_gh(self):
        # load het. functions h if they exist. otherwise generate.
        
        if self.recompute_gh or not(lib.files_exist(self.ghx_fnames,self.ghy_fnames)):
            self.generate_gh()  # populate ghx_list, ghy_list

            for i in range(self.trunc_g+1):
                dill.dump(self.ghx_list[i],open(self.ghx_fnames[i],'wb'),recurse=True)
                dill.dump(self.ghy_list[i],open(self.ghy_fnames[i],'wb'),recurse=True)

        else:
            self.ghx_list = lib.load_dill(self.ghx_fnames)
            self.ghy_list = lib.load_dill(self.ghy_fnames)
            
    def generate_gh(self):
        """
        generate heterogeneous terms for the Floquet eigenfunctions g.

        Returns
        -------
        list of symbolic heterogeneous terms in self.ghx_list, self.ghy_list.

        """
        
        self.ghx_list = []
        self.ghy_list = []
        
        # get the general expression for h before plugging in g.
        self.hx = 0
        self.hy = 0
        
        for i in range(2,self.trunc_gh+1):
            # all x1,x2 are evaluated on limit cycle x=cos(t), y=sin(t)
            p = lib.kProd(i,self.dx)
            d1 = lib.vec(lib.df(self.NIC1,self.x,i))
            d2 = lib.vec(lib.df(self.NIC2,self.x,i))
            
            self.hx += (1/math.factorial(i)) * np.dot(p,d1)
            self.hy += (1/math.factorial(i)) * np.dot(p,d2)
            
        self.hx = sym.Matrix(self.hx)
        self.hy = sym.Matrix(self.hy)
        
        # expand all terms
        self.hx = sym.expand(self.hx.subs([(self.dx1,self.gx),(self.dx2,self.gy)]))
        self.hy = sym.expand(self.hy.subs([(self.dx1,self.gx),(self.dx2,self.gy)]))
        
        # collect all psi terms into list of some kind
        self.tempx = sym.collect(self.hx[0],self.psi,evaluate=False)
        self.tempy = sym.collect(self.hy[0],self.psi,evaluate=False)
        
        counter = 0
        while (counter <= self.trunc_g+1):
            # save current term
            self.ghx_list.append(self.tempx[self.psi**counter])
            self.ghy_list.append(self.tempy[self.psi**counter])
        
            counter += 1
            
        # substitute limit cycle. maybe move elsewhere.
        for i in range(len(self.ghx_list)):
            self.ghx_list[i] = self.ghx_list[i].subs({self.x1:sym.cos(2*sym.pi*self.t),
                                                      self.x2:sym.sin(2*sym.pi*self.t),
                                                      sym.Indexed('gx',0):s(0),
                                                      sym.Indexed('gy',0):s(0)})
            self.ghy_list[i] = self.ghy_list[i].subs({self.x1:sym.cos(2*sym.pi*self.t),
                                                      self.x2:sym.sin(2*sym.pi*self.t),
                                                      sym.Indexed('gx',0):s(0),
                                                      sym.Indexed('gy',0):s(0)})
                    
    def load_g(self):
        
        # load all g or recompute or compute new.
        if self.recompute_g or not(lib.files_exist(self.g_fnames)):  # generate
            self.g_list, self.gx_list, self.gy_list = ([] for i in range(3))

            print('* Computing...',end=' ')
            
            for i in range(self.trunc_g+1):
                print('g_'+str(i),end=', ')
                self.generate_g(i)
            print()
            
            # save
            for i in range(self.trunc_g+1): 
                np.savetxt(self.g_fnames[i],self.g_list[i])

        else:   # load
            
            self.g_list,self.gx_list,self.gy_list = self.load_sols(self.g_fnames,name='g')
            
        rulex = {sym.Indexed('gx',i):self.gx_list[i](self.t) for i in range(len(self.g_fnames))}
        ruley = {sym.Indexed('gy',i):self.gy_list[i](self.t) for i in range(len(self.g_fnames))}
        self.rule_g = {**rulex,**ruley}

    def generate_g(self,k,total_iter=4):
        # load kth expansion of g for k >= 0
        
        if k == 0:
            # g0 is 0
            
            self.g_list.append(np.zeros((self.TN,2)))
            self.gx_list.append(implemented_function('gx_0', lambda t: 0))
            self.gy_list.append(implemented_function('gy_0', lambda t: 0))

            return
        
        rulex = {sym.Indexed('gx',i):self.gx_list[i](self.t) for i in range(k)}
        ruley = {sym.Indexed('gy',i):self.gy_list[i](self.t) for i in range(k)}
        rule = {**rulex,**ruley}
        
        # apply replacement 
        self.ghx_list[k] = self.ghx_list[k].subs(rule)
        self.ghy_list[k] = self.ghy_list[k].subs(rule)
        
        # lambdify heterogeneous terms for use in integration
        hetx_lam = lambdify(self.t,self.ghx_list[k])
        hety_lam = lambdify(self.t,self.ghy_list[k])
        
        
        
        self.method = 'RK45'
        self.rtol = 1e-4
        self.atol = 1e-8
        
        # find intial condtion
        if k == 1:
            #init = [0,0]
            init = copy.deepcopy(self.g1_init)
            total_iter = 1
        else:
            init = [0,0]
            # Newton
            for mm in range(total_iter):
                out = lib.get_newton_jac(self,self.dg,-self.tLC,init,hetx_lam,hety_lam,k)
                print(out)
                init += out
        
        # get full solution
        gu = odeint(self.dg,init,-self.tLC,args=(hetx_lam,hety_lam,k),tfirst=True)

        gu = gu[::-1,:]
        # save soluton as lambda functions
        self.g_list.append(gu)
        
        fnx = interp1d(self.tLC,gu[:,0],fill_value='extrapolate')
        fny = interp1d(self.tLC,gu[:,1],fill_value='extrapolate')
        
        self.gx_list.append(implemented_function('gx_'+str(k),self.myFun(fnx)))
        self.gy_list.append(implemented_function('gy_'+str(k),self.myFun(fny)))
        
        if True and k == 1:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.tLC,gu)
            ax.set_title('g1')
            plt.show(block=True)
            
        if True and k == 2:
            t = np.linspace(0,1,100)
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #ax.plot(t,hetx_lam(t))
            #ax.set_title('hetx_lam')
            #plt.show(block=True)
            
            
            fn = lambdify(self.t,self.gx_list[1](self.t))
            
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(t,fn(t))
            y = self.g_list[1][:,0]
            ax.plot(np.linspace(0,1,len(y)),y)
            ax.set_title('gx_list[1]')
            plt.show(block=True)

        
    def load_sols(self,fnames,name='g'):
        """
        

        Parameters
        ----------
        fnames : list
            list of file names. single file names should be intered as [fname]
        name : str, optional
            load solutions g, z, or i. The default is 'g'.

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
            
            fnx = interp1d(self.tLC,data[:,0])
            fny = interp1d(self.tLC,data[:,1])
        
            xtemp = implemented_function(name+'x_'+str(i), self.myFun(fnx))
            ytemp = implemented_function(name+'y_'+str(i), self.myFun(fny))
            
            listx.append(xtemp)
            listy.append(ytemp)
            
        return list1, listx, listy
    

    
    def load_het(self):
        # load het. functions h if they exist. otherwise generate.
        
        if self.recompute_het or \
                not(lib.files_exist(self.hetx_fnames,self.hety_fnames,[self.A_fname])):
            self.generate_het()
            
            # save matrix of a_i
            dill.dump(self.A,open(self.A_fname,'wb'),recurse=True)
            
            # save het. terms
            for i in range(self.trunc_g+1):
                dill.dump(self.hetx_list[i],open(self.hetx_fnames[i],'wb'),recurse=True)
                dill.dump(self.hety_list[i],open(self.hety_fnames[i],'wb'),recurse=True)

        else:
            self.A, = lib.load_dill([self.A_fname])
            self.hetx_list = lib.load_dill(self.hetx_fnames)
            self.hety_list = lib.load_dill(self.hety_fnames)
            
            
    def generate_het(self):
        """
        Generate heterogeneous terms for integrating the Z_i and I_i terms.

        Returns
        -------
        None.

        """
        
        self.hetx_list, self.hety_list = ([] for i in range(2))
        # get the general expression for h in z before plugging in g,z.
        
        # column vectors ax ay for use in matrix A = [ax ay]
        self.ax = Matrix([[0],[0]])
        self.ay = Matrix([[0],[0]])
        
        for j in range(1,self.trunc_gh+1):
            p1 = lib.kProd(j,self.dx)
            p2 = kp(p1,sym.eye(2))
        
            d1 = lib.vec(lib.df(self.NIC1,self.x,j+1))
            d2 = lib.vec(lib.df(self.NIC2,self.x,j+1))
            
            self.ax += (1/math.factorial(j)) * p2*d1
            self.ay += (1/math.factorial(j)) * p2*d2
        
        self.A = sym.zeros(2,2)
        
        self.A[:,0] = self.ax
        self.A[:,1] = self.ay
        
        
        self.z_expansion = Matrix([[self.zx],[self.zy]])
        het = self.A*self.z_expansion
        
        
        # expand all terms
        self.hetx = sym.expand(het[0].subs([(self.dx1,self.gx),(self.dx2,self.gy)]))
        self.hety = sym.expand(het[1].subs([(self.dx1,self.gx),(self.dx2,self.gy)]))
        
        # collect all psi terms into factors of pis^k
        self.hetx_powers = sym.collect(self.hetx,self.psi,evaluate=False)
        self.hety_powers = sym.collect(self.hety,self.psi,evaluate=False)
    
    
        self.hetx_list = []
        self.hety_list = []
        
        counter = 0
        while (counter <= self.trunc_g+1):
            
            # save current term
            self.hetx_list.append(self.hetx_powers[self.psi**counter])
            self.hety_list.append(self.hety_powers[self.psi**counter])
        
            counter += 1
        
        # substitute limit cycle
        for i in range(len(self.ghx_list)):
            self.hetx_list[i] = self.hetx_list[i].subs({self.x1:sym.cos(2*sym.pi*self.t),
                                                        self.x2:sym.sin(2*sym.pi*self.t),
                                                        sym.Indexed('gx',0):s(0),
                                                        sym.Indexed('gy',0):s(0)})
            self.hety_list[i] = self.hety_list[i].subs({self.x1:sym.cos(2*sym.pi*self.t),
                                                        self.x2:sym.sin(2*sym.pi*self.t),
                                                        sym.Indexed('gx',0):s(0),
                                                        sym.Indexed('gy',0):s(0)})
         
            
    def load_z(self):
        
        # load all g or recompute or compute new.
        self.z_list, self.zx_list, self.zy_list = ([] for i in range(3))
        self.zx_callable, self.zy_callable = ([] for i in range(2))

        if self.recompute_z or not(lib.files_exist(self.z_fnames)):
            
            print('* Computing...',end=' ')
            for i in range(self.trunc_g+1):
                print('z_'+str(i), end=', ')
                self.z_list.append(self.generate_z(i))
                np.savetxt(self.z_fnames[i],self.z_list[i])
            print()
            
        else:
            self.z_list, self.zx_list, self.zy_list = self.load_sols(self.z_fnames,name='z')
            
        for i in range(len(self.z_list)):
            
            self.zx_callable.append(lambdify(self.t,self.zx_list[i](self.t)))
            self.zy_callable.append(lambdify(self.t,self.zy_list[i](self.t)))
 
        
            
    def generate_z(self,k,total_iter=5):
        
        # load kth expansion of g for k >= 1
        rulex = {sym.Indexed('zx',i):self.zx_list[i](self.t) for i in range(k)}
        ruley = {sym.Indexed('zy',i):self.zy_list[i](self.t) for i in range(k)}
        rule = {**rulex,**ruley,**self.rule_g}
        
        zhx = self.hetx_list[k].subs(rule)
        zhy = self.hety_list[k].subs(rule)
        
        hetx_lam = lambdify(self.t,zhx)
        hety_lam = lambdify(self.t,zhy)
        
        
        if k == 0:
            init = copy.deepcopy(self.z0_init)
            total_iter = 1
        else:
            init = [0,0]
        
            # Newton
            for mm in range(total_iter):
                init += lib.get_newton_jac(self.dz,self.tLC,init,hetx_lam,hety_lam,k)
            
        zu = odeint(self.dz,init,self.tLC,args=(hetx_lam,hety_lam,k),tfirst=True)
        
        if k == 0:
            # normalize
            dLC = lib.rhs([np.cos(0),np.sin(0)],0,self.f)
            print('dLC',dLC,'zu[0,:]',zu[0,:],'z0 init',self.z0_init,'np.dot(dLC,zu[0,:])',np.dot(dLC,zu[0,:]))
            
            zu = self.omega*zu/(np.dot(dLC,zu[0,:]))
        
        
        fnx = interp1d(self.tLC,zu[:,0])
        fny = interp1d(self.tLC,zu[:,1])
        self.zx_list.append(implemented_function('zx_'+str(k), self.myFun(fnx)))
        self.zy_list.append(implemented_function('zy_'+str(k), self.myFun(fny)))
        
        self.zx_callable.append(lambdify(self.t,self.zx_list[k](self.t)))
        self.zy_callable.append(lambdify(self.t,self.zy_list[k](self.t)))
        
        return zu
    

    def load_i(self):
        
        # load all g or recompute or compute new.
        self.i_list, self.ix_list, self.iy_list = ([] for i in range(3))
        self.ix_callable, self.iy_callable = ([] for i in range(2))

        if self.recompute_i or not(lib.files_exist(self.i_fnames)):
            
            print('* Computing...',end=' ')
            for i in range(self.trunc_g+1):
                print('i_'+str(i), end=', ')
                self.i_list.append(self.generate_i(i))

                

                np.savetxt(self.i_fnames[i],self.i_list[i])
            print()
            
        else:
            self.i_list, self.ix_list, self.iy_list = self.load_sols(self.i_fnames,name='i')
            
        
        for i in range(len(self.i_list)):
            self.ix_callable.append(lambdify(self.t,self.ix_list[i](self.t)))
            self.iy_callable.append(lambdify(self.t,self.iy_list[i](self.t)))
    
    def generate_i(self,k,total_iter=5):
        
        # load kth expansion of g for k >= 1
        rulex = {sym.Indexed('zx',i):self.ix_list[i](self.t) for i in range(k)}
        ruley = {sym.Indexed('zy',i):self.iy_list[i](self.t) for i in range(k)}
        rule = {**rulex,**ruley,**self.rule_g}

        ihx = self.hetx_list[k].subs(rule)
        ihy = self.hety_list[k].subs(rule)
        
        hetx_lam = lambdify(self.t,ihx)
        hety_lam = lambdify(self.t,ihy)
        
        
        if k == 0:
            init = copy.deepcopy(self.i0_init)
            total_iter = 0
        else:
            init = [0,0]
        
            # Newton
            for mm in range(total_iter):
                
                if False and k == 1:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    
                    iu = odeint(self.di,init,self.tLC,args=(hetx_lam,hety_lam,k),tfirst=True)

                    ax.plot(iu[:,0])
                    ax.plot(iu[:,1])
                    
                    ax.set_title('mm='+str(mm)+', k='+str(k))
                    
                    plt.show(block=True)
                    
                    
                    
                init += lib.get_newton_jac(self.di,self.tLC,init,hetx_lam,hety_lam,k)
                
        iu = odeint(self.di,init,self.tLC,args=(hetx_lam,hety_lam,k),tfirst=True)
        
        if k == 1:  # normalize
            
            gx = lambdify(self.t,self.gx_list[1](self.t))
            gy = lambdify(self.t,self.gy_list[1](self.t))
            
            zx = lambdify(self.t,self.zx_list[0](self.t))
            zy = lambdify(self.t,self.zy_list[0](self.t))
            
            ix = lambdify(self.t,self.ix_list[0](self.t))
            iy = lambdify(self.t,self.iy_list[0](self.t))
            
            F = lib.rhs([np.cos(0),np.sin(0)],0,self.f)
            g1 = np.array([gx(0),gy(0)])
            z0 = np.array([zx(0),zy(0)])
            i0 = np.array([ix(0),iy(0)])
            
            J = self.jacLC(0)
            i1 = iu[0,:]
            
            ijg = np.dot(i0,np.dot(J,g1))
            be = (self.kappa - ijg - np.dot(i1,F))/(np.dot(z0,F))
            
            init = iu[0,:] + be*z0
            iu = odeint(self.di,init,self.tLC,args=(hetx_lam,hety_lam,k),tfirst=True)
        
    
        fnx = interp1d(self.tLC,iu[:,0])
        fny = interp1d(self.tLC,iu[:,1])
        self.ix_list.append(implemented_function('ix_'+str(k), self.myFun(fnx)))
        self.iy_list.append(implemented_function('iy_'+str(k), self.myFun(fny)))
        
        self.ix_callable.append(lambdify(self.t,self.ix_list[k](self.t)))
        self.iy_callable.append(lambdify(self.t,self.iy_list[k](self.t)))
        
        return iu


        
    def interp(self,t,data):
        """
        interp1d function is created each time on function call so it is slow
        but it is also easily generalizable for any T-periodic function
        
        Parameters
        ----------
        t : time
        data : 1d array of x or y coordinate from integration.
    
        Returns
        -------
        interporated function
    
        """
        # helps to keep mod T here for integrators
        t = np.mod(t,self.T)
        fn = interp1d(self.tLC,data)
        
        #print('t,fn',t,fn(t))
            
        return fn(t)
    
    def dg(self,t,z,hetx,hety,order):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        #if order == 2:
        #    print(t,hetx(t))
        
        jac = self.jacLC(t)*(order > 0)
        
        return np.dot(jac-order*self.kappa*self.eye,z) + 0.5*np.array([hetx(t),hety(t)])
    
    def dz(self,t,z,hetx,hety,order):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        return -np.dot(self.jacLC(t).T+order*self.kappa*self.eye,z) - np.array([hetx(t),hety(t)])
    
    def di(self,t,z,hetx,hety,order):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = -np.dot(self.jacLC(t).T+self.kappa*(order-1)*self.eye,z)
        het = -np.array([hetx(t),hety(t)])
        
        out = hom+het
        
        return out
    
    def myFun(self,fn):
        return lambda x=self.t: fn(np.mod(x,self.T))
    
    def plot(self,option='g1'):
        
        # check if option of the form 'g'+'int'
        if (option[0] == 'g') and (option[1:].isnumeric()):
            k = int(option[1:])
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.tLC,self.g_list[k][:,0])
            ax.plot(self.tLC,self.g_list[k][:,1])
            ax.set_title(option)
            
            print('g init',k,self.g_list[k][0,:])
            
        if (option[0] == 'z') and (option[1:].isnumeric()):
            k = int(option[1:])
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.tLC,self.z_list[k][:,0])
            ax.plot(self.tLC,self.z_list[k][:,1])
            ax.set_title(option)
            
            print('z init',k,self.z_list[k][0,:])

        # check if option of the form 'g'+'int'
        if (option[0] == 'i') and (option[1:].isnumeric()):
            k = int(option[1:])
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.tLC,self.i_list[k][:,0])
            ax.plot(self.tLC,self.i_list[k][:,1])
            ax.set_title(option)
            
            print('i init',k,self.i_list[k][0,:])
            
        if option == 'surface_z':
            fig = plt.figure(figsize=(4,4))
            ax = fig.gca(projection='3d')
            
            # Make data.
            th = np.arange(0, 1, .01)
            psi = np.arange(-1, 1, .01)
            
            th, psi = np.meshgrid(th, psi)
            
            Z = 0
            for i in range(self.trunc_g+1):
                Z += psi**i*self.zx_callable[i](th)
            #Z = self.zx_callable[0](th)+psi*self.zx_callable[1](th)

            # Plot the surface.
            ax.plot_surface(th, psi, Z,cmap='viridis')
            ax.view_init(15,-45)
            
            """
            X = np.arange(-5, 5, 0.25)
            Y = np.arange(-5, 5, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X**2 + Y**2)
            Z = np.sin(R)
            
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z,
                       linewidth=0, antialiased=False)
            """

            
        if option == 'surface_i':
            fig = plt.figure(figsize=(4,4))
            ax = fig.gca(projection='3d')
            
            # Make data.
            th = np.arange(0, 1, .01)
            psi = np.arange(-1, 1, .01)
            
            th, psi = np.meshgrid(th, psi)
            
            Z = 0
            for i in range(self.trunc_g+1):
                Z += psi**i*self.ix_callable[i](th)
            #Z = self.zx_callable[0](th)+psi*self.zx_callable[1](th)

            # Plot the surface.
            ax.plot_surface(th, psi, Z,cmap='viridis')
            ax.view_init(15,-45)
            
            """
            X = np.arange(-5, 5, 0.25)
            Y = np.arange(-5, 5, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X**2 + Y**2)
            Z = np.sin(R)
            
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z,
                       linewidth=0, antialiased=False)
            """


def main():
    
    # for NIC, 3rd derivatives go away, so we only need trunc_gh=3.
    a = NIC(recompute_gh=False,
            recompute_g=True,
            recompute_het=False,
            recompute_z=False,
            recompute_i=False,
            trunc_g=3,trunc_gh=4)
    
    
    for i in range(a.trunc_g+1):
        a.plot('g'+str(i))
        
    for i in range(a.trunc_g+1):
        a.plot('z'+str(i))
        
    for i in range(a.trunc_g+1):
        a.plot('i'+str(i))
        
    
    a.plot('surface_z')
    a.plot('surface_i')
    
    plt.show(block=True)
    
    
if __name__ == "__main__":
    main()