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
#import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft
from operator import methodcaller
from mpl_toolkits.mplot3d import Axes3D
from sympy import Matrix, symbols,diff, pi, Sum, Indexed, collect, expand
#from sympy import Function
from sympy import sympify as s
from sympy.physics.quantum import TensorProduct as kp
from sympy.utilities.lambdify import lambdify, implemented_function
#from sympy.interactive import init_printing

from scipy.interpolate import interp1d, interp2d
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
            'q_val':2*np.pi,
            'eps_val':0,
            'd_val':1,
            
            'trunc_order':3,
            'trunc_derivative':2,
            
            'TN':10000,
            'dir':'dat',
            
            'recompute_monodromy':True,
            'recompute_gh':False,
            'recompute_g':False,
            'recompute_het':False,
            'recompute_z':False,
            'recompute_i':False,
            'recompute_p':False,
            'recompute_h_odd':False,
            }
        
        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # misc variables
        self.miter = self.trunc_order+1

        # Symbolic variables and functions
        self.eye = np.identity(2)
        self.psi, self.eps, self.kap_sym = sym.symbols('psi eps kap_sym')
        
        
        #self.x1,self.x2,self.x3,self.t = symbols('x1 x2,x3, t')
        #self.x,self.y,self.z,self.t = symbols('x1 x2,x3, t')
        #self.f_list, self.x_vars, self.y_vars, self.z_vars = ([] for i in range(4))
        
        # single-oscillator variables
        self.x, self.y, self.q, self.t, self.s = symbols('x y q t s')
        self.tA, self.tB, = symbols('tA tB')
        self.dx, self.dy = symbols('dx dy')
        
        # coupling variables
        self.thA, self.psiA, self.thB, self.psiB = symbols('thA psiA thB psiB')
        
        self.xA, self.yA, self.xB, self.yB = symbols('xA yA xB yB')
        self.dxA, self.dyA, self.dxB, self.dyB = symbols('dxA dyA dxB dyB')
        
        self.dx_vec = Matrix([[self.dx,self.dy]])
        self.x_vec = Matrix([[self.x],[self.y]])
        
        # find limit cycle or load
        self.T = 2*np.pi/self.q_val
        
        #self.TN = TN
        # find limit cycle -- easy for this problem but think of general methods
        # see Dan's code for general method
        self.tLC = np.linspace(0,self.T,self.TN)
        
        # make interpolated version of LC
        #LC_interp_x = interp1d(tLC,LC_arr[:,0])
        #LC_interp_y = interp1d(tLC,LC_arr[:,1])
        
        self.LC_x = sym.cos(self.q_val*self.t)
        self.LC_y = sym.sin(self.q_val*self.t)
        
        self.LC_x_sym = sym.cos(self.q*self.t)
        self.LC_y_sym = sym.sin(self.q*self.t)
        
        self.rule_LC = {self.x:self.LC_x,self.y:self.LC_y}
        
        # filenames and directories
        self.dir = 'cgl_dat/'
        
        if (not os.path.exists(self.dir)):
            os.makedirs(self.dir)
        
        self.generate_fnames()
        self.CGL_sym = self.CGL_rhs(0,[self.x,self.y],option='sym')
        
        # symbol J on LC.
        self.jacLC_sym = Matrix([[diff(self.CGL_sym[0],self.x),diff(self.CGL_sym[0],self.y)],
                                 [diff(self.CGL_sym[1],self.x),diff(self.CGL_sym[1],self.y)]])
        
        self.jacLC_sym = self.jacLC_sym.subs({self.x:self.LC_x,self.y:self.LC_y})
        
        
        # make RHS and Jacobian callable functions
        self.f = lambdify((self.x,self.y),self.CGL_sym.subs({'q':self.q_val}))
        
        #jac = sym.lambdify((x1,x2),Jac)
        self.jacLC = lambdify((self.t),self.jacLC_sym)
        
        # assume gx is the first coordinate of Floquet eigenfunction g
        # brackets denote Taylor expansion functions
        # now substitute Taylor expansion dx = gx[0] + gx[1] + gx[2] + ...
        self.generate_reduced_expansions()
        
        # Run method
        # get monodromy matrix
        self.load_monodromy()

        self.load_gh()  # get heterogeneous terms for g
        self.load_g()  # get g
        
        #t0 = time.time()
        self.load_het()
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
        
        self.generate_coupling_expansions()
        self.generate_ch()
        
        self.load_ph()
        self.load_p()
        
        self.load_h()
        
    def CGL_rhs(self,t,z,option='value'):
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
            return np.array([x*(1-R2)-self.q_val*R2*y,y*(1-R2)+self.q_val*R2*x])
        elif option == 'sym':
            return Matrix([x*(1-R2)-self.q_val*R2*y,y*(1-R2)+self.q_val*R2*x])
    
    
    def CGL_coupling(self,x1,y1,x2,y2,option='value'):
        """
        r^(2n) to r^n function. default parameter order is from perspective of
        first oscillator.
        
        in this case the input is (x1,y1,x2,y2) and the output is an r2 vec.
        """
        
        if option == 'value':
            return np.array([x2-x1-self.d_val*(y2-y1),y2-y1+self.d_val*(x2-x1)])
        elif option == 'sym':
            return Matrix([x2-x1-self.d*(y2-y1),y2-y1+self.d*(x2-x1)])
    
        return 
        

    
    def generate_fnames(self):
        
        self.model_params = '_q=' + str(self.q_val)
        
        self.sim_params = '_TN='+str(self.TN)
        
        self.monodromy_fname = self.dir+'monodromy_'+self.model_params+self.sim_params+'.txt'

        self.ghx_fnames = [self.dir+'ghx_'+str(i)+self.model_params+'.d' 
                           for i in range(self.miter)]
        self.ghy_fnames = [self.dir+'ghy_'+str(i)+self.model_params+'.d' 
                           for i in range(self.miter)]
        self.g_fnames = [self.dir+'g_'+str(i)+self.model_params+self.sim_params+'.txt' 
                         for i in range(self.miter)]
        
        self.hetx_fnames = [self.dir+'hetx_'+str(i)+self.model_params+'.d' 
                            for i in range(self.miter)]
        self.hety_fnames = [self.dir+'hety_'+str(i)+self.model_params+'.d' 
                            for i in range(self.miter)]
        self.A_fname = self.dir+'A_'+self.model_params+'.d'
        
        self.z_fnames = [self.dir+'z_'+str(i)+self.model_params+self.sim_params+'.txt' 
                         for i in range(self.miter)]
        self.i_fnames = [self.dir+'i_'+str(i)+self.model_params+self.sim_params+'.txt' 
                         for i in range(self.miter)]
        
        self.pA_fnames = [self.dir+'pA_'+str(i)+self.model_params+'.txt' 
                          for i in range(self.miter)]
        self.pB_fnames = [self.dir+'pB_'+str(i)+self.model_params+'.txt' 
                          for i in range(self.miter)]
        
        self.h_odd_fnames = [self.dir+'h_odd_'+str(i)+self.model_params+'.txt' 
                             for i in range(self.miter)]
    
    def generate_reduced_expansions(self):
        """
        generate expansions from Wilson 2020
        """
        i_sym = sym.symbols('i_sym')  # summation index
        psi = self.psi
        
        self.gx = Sum(psi**i_sym*Indexed('gx',i_sym),(i_sym,0,self.miter)).doit()
        self.gy = Sum(psi**i_sym*Indexed('gy',i_sym),(i_sym,0,self.miter)).doit()
        
        self.zx = Sum(psi**i_sym*Indexed('zx',i_sym),(i_sym,0,self.miter)).doit()
        self.zy = Sum(psi**i_sym*Indexed('zy',i_sym),(i_sym,0,self.miter)).doit()
        
        self.ix = Sum(psi**i_sym*Indexed('ix',i_sym),(i_sym,0,self.miter)).doit()
        self.iy = Sum(psi**i_sym*Indexed('iy',i_sym),(i_sym,0,self.miter)).doit()
        
        self.z_expansion = Matrix([[self.zx],[self.zy]])
        
    def generate_coupling_expansions(self):
        """
        generate expansions for coupling.
        """
        
        i_sym = sym.symbols('i_sym')  # summation index
        psi = self.psi
        eps = self.eps
        
        self.pA = Sum(eps**i_sym*Indexed('pA',i_sym),(i_sym,1,self.miter)).doit()
        self.pB = Sum(eps**i_sym*Indexed('pB',i_sym),(i_sym,1,self.miter)).doit()
        
        gxA = Sum(psi**i_sym*Indexed('gxA',i_sym),(i_sym,1,self.miter)).doit()
        gyA = Sum(psi**i_sym*Indexed('gyA',i_sym),(i_sym,1,self.miter)).doit()
        ixA = Sum(psi**i_sym*Indexed('ixA',i_sym),(i_sym,0,self.miter)).doit()
        iyA = Sum(psi**i_sym*Indexed('iyA',i_sym),(i_sym,0,self.miter)).doit()
        
        gxB = Sum(psi**i_sym*Indexed('gxB',i_sym),(i_sym,1,self.miter)).doit()
        gyB = Sum(psi**i_sym*Indexed('gyB',i_sym),(i_sym,1,self.miter)).doit()
        ixB = Sum(psi**i_sym*Indexed('ixB',i_sym),(i_sym,0,self.miter)).doit()
        iyB = Sum(psi**i_sym*Indexed('iyB',i_sym),(i_sym,0,self.miter)).doit()
        
        ruleA = {'psi':self.pA}
        ruleB = {'psi':self.pB}
        
        gx_collectedA = collect(expand(gxA.subs(ruleA)),eps)
        gy_collectedA = collect(expand(gyA.subs(ruleA)),eps)
        ix_collectedA = collect(expand(ixA.subs(ruleA)),eps)
        iy_collectedA = collect(expand(iyA.subs(ruleA)),eps)
        
        gx_collectedB = collect(expand(gxB.subs(ruleB)),eps)
        gy_collectedB = collect(expand(gyB.subs(ruleB)),eps)
        ix_collectedB = collect(expand(ixB.subs(ruleB)),eps)
        iy_collectedB = collect(expand(iyB.subs(ruleB)),eps)
        
        # truncate and collect up to order self.trunc_order
        self.gx_epsA = 0
        self.gy_epsA = 0
        self.ix_epsA = 0
        self.iy_epsA = 0
        
        self.gx_epsB = 0
        self.gy_epsB = 0
        self.ix_epsB = 0
        self.iy_epsB = 0
        
        for i in range(self.miter):
            self.gx_epsA += eps**i*gx_collectedA.coeff(eps,i)
            self.gy_epsA += eps**i*gy_collectedA.coeff(eps,i)
            self.ix_epsA += eps**i*ix_collectedA.coeff(eps,i)
            self.iy_epsA += eps**i*iy_collectedA.coeff(eps,i)
            
            self.gx_epsB += eps**i*gx_collectedB.coeff(eps,i)
            self.gy_epsB += eps**i*gy_collectedB.coeff(eps,i)
            self.ix_epsB += eps**i*ix_collectedB.coeff(eps,i)
            self.iy_epsB += eps**i*iy_collectedB.coeff(eps,i)
    
        self.iA = Matrix([[self.ix_epsA],[self.iy_epsA]])
        self.iB = Matrix([[self.ix_epsB],[self.iy_epsB]])
        
    
    def generate_ch(self):
        
        # find K_i^{j,k}
        coupA = self.CGL_coupling(*self.x_pairA)
        coupB = self.CGL_coupling(*self.x_pairB)
        
        cx_symA = coupA[0]
        cy_symA = coupA[1]
        
        cx_symB = coupB[0]
        cy_symB = coupB[1]
        
        cxA = cx_symA + lib.df(cx_symA,self.x_pairA,1).dot(self.dx_pairA)
        cyA = cy_symA + lib.df(cy_symA,self.x_pairA,1).dot(self.dx_pairA)
        
        cxB = cx_symB + lib.df(cx_symB,self.x_pairB,1).dot(self.dx_pairB)
        cyB = cy_symB + lib.df(cy_symB,self.x_pairB,1).dot(self.dx_pairB)
        
        # get expansion for coupling term
        for i in range(2,self.trunc_derivative+1):
            # all x1,x2 are evaluated on limit cycle x=cos(t), y=sin(t)
            kA = lib.kProd(i,self.dx_pairA)
            dxA = lib.vec(lib.df(cx_symA,self.x_pairA,i))
            dyA = lib.vec(lib.df(cy_symA,self.x_pairA,i))
            
            cxA += (1/math.factorial(i)) * kA.dot(dxA)
            cyA += (1/math.factorial(i)) * kA.dot(dyA)
    
            kB = lib.kProd(i,self.dx_pairB)
            dxB = lib.vec(lib.df(cx_symB,self.x_pairB,i))
            dyB = lib.vec(lib.df(cy_symB,self.x_pairB,i))
            
            cxB += (1/math.factorial(i)) * kB.dot(dxB)
            cyB += (1/math.factorial(i)) * kB.dot(dyB)
    
        
        rule = {'dxA':self.gx_epsA,'dyA':self.gy_epsA,
                'dxB':self.gx_epsB,'dyB':self.gy_epsB}
        
        self.cxA = cxA.subs(rule)
        self.cyA = cyA.subs(rule)

        self.cxB = cxB.subs(rule)
        self.cyB = cyB.subs(rule)
        
        self.cA = Matrix([[self.cxA],[self.cyA]])
        self.cB = Matrix([[self.cxB],[self.cyB]])
        
        # now put in powers of eps
        self.KxA = []
        self.KyA = []
        
        self.KxB = []
        self.KyB = []
        
        cxA_collected = collect(expand(self.cxA),self.eps)
        cyA_collected = collect(expand(self.cyA),self.eps)
        
        cxB_collected = collect(expand(self.cxB),self.eps)
        cyB_collected = collect(expand(self.cyB),self.eps)
        
        # collect and store into K.
        for i in range(self.miter):
            
            eps_i_termxA = cxA_collected.coeff(self.eps,i)
            eps_i_termyA = cyA_collected.coeff(self.eps,i)
            
            eps_i_termxB = cxB_collected.coeff(self.eps,i)
            eps_i_termyB = cyB_collected.coeff(self.eps,i)
            
            #cA_order = Matrix([[eps_i_termxA],[eps_i_termyA]])
            #cB_order = Matrix([[eps_i_termxB],[eps_i_termyB]])
            
            self.KxA.append(eps_i_termxA)
            self.KyA.append(eps_i_termyA)
            
            self.KxB.append(eps_i_termxB)
            self.KyB.append(eps_i_termyB)

            #print(cA_order)

    def load_ph(self):
        """
        generate/load the het. terms for psi ODEs.
        """
        
        self.generate_ph()
        
    def generate_ph(self):
        # rename for shortness
        thA = self.thA
        thB = self.thB
        
        #gxs, gys, ixs, iys = (self.gx_imp, self.gy_imp,
        #                      self.ix_imp, self.iy_imp)
        
        gxs, gys, ixs, iys = (self.gx_imp, self.gy_imp,
                              self.ix_imp, self.iy_imp)
        
        # collect left and right hand terms
        ircA = self.kap_sym*self.pA + self.eps*self.iA.dot(self.cA)
        ircA = collect(expand(ircA),self.eps)
        
        ircB = self.kap_sym*self.pB + self.eps*self.iB.dot(self.cB)
        ircB = collect(expand(ircB),self.eps)

        # create rule for replacing with interpolated functions
        rule_LC = {'xA':self.LC_xA,'yA':self.LC_yA,
                   'xB':self.LC_xB,'yB':self.LC_yB}
        
        self.NA = 50
        self.NB = 51
        self.A_array = np.linspace(0,1,self.NA)
        self.B_array = np.linspace(0,1,self.NB)
        
        self.A_mg, self.B_mg = np.meshgrid(self.A_array,self.B_array)
        
        #print(self.A_mg[0,:])
        #print(self.B_mg[0,:])
        
        rule_ixA = {Indexed('ixA',i):ixs[i](thA) for i in range(self.miter)}
        rule_iyA = {Indexed('iyA',i):iys[i](thA) for i in range(self.miter)}
        
        rule_ixB = {Indexed('ixB',i):ixs[i](thB) for i in range(self.miter)}
        rule_iyB = {Indexed('iyB',i):iys[i](thB) for i in range(self.miter)}
        
        rule = {**rule_LC,
                **self.rule_g_AB,
                **rule_ixA,**rule_iyA,
                **rule_ixB,**rule_iyB}
    
        # get list of coeffs up to self.trunc_order order
        self.p_rhsA = []
        self.p_rhsB = []
        
        self.ph_symA = []
        self.ph_symB = []
        
        
        for i in range(self.miter):
            eps_i_termA = ircA.coeff(self.eps,i)
            eps_i_termA = eps_i_termA.subs(rule)
            
            eps_i_termB = ircB.coeff(self.eps,i)
            eps_i_termB = eps_i_termB.subs(rule)
            
            self.p_rhsA.append(eps_i_termA)
            self.ph_symA.append(eps_i_termA - self.kap_sym*Indexed('pA',i))
            
            self.p_rhsB.append(eps_i_termB)
            self.ph_symB.append(eps_i_termB - self.kap_sym*Indexed('pB',i))
            
            #print()
            #print(self.ph_symA[i])
        
        
    def load_p(self):
        """
        generate/load the ODEs for psi.
        """
        
        self.interval = np.linspace(0,5,100)
        self.ds = (self.interval[-1]-self.interval[0])/len(self.interval)
        self.dxA = (self.A_array[-1]-self.A_array[0])/len(self.A_array)
        self.dxB = (self.B_array[-1]-self.B_array[0])/len(self.B_array)
        
        # load all p or recompute or compute new.
        self.pA_data, self.pB_data, self.pA_imp, self.pB_imp = ([] for i in range(4))
        self.pA_callable, self.pB_callable = ([] for i in range(2))

        # generate
        if self.recompute_p or not(lib.files_exist(self.pA_fnames,self.pB_fnames)):
            print('* Computing...',end=' ')
            for i in range(self.miter):
                print('p_'+str(i),end=', ')
                self.generate_p(i)
            print()
            
            # save
            for i in range(self.miter): 
                if False:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    ax.plot_surface(self.A_mg,self.B_mg,
                                    self.pA_data[i],
                                    cmap='viridis')
                    
                    ax.set_title('plot from save')
                
                print('genmerated pa shape',np.shape(self.pA_data[i]))
                np.savetxt(self.pA_fnames[i],self.pA_data[i])
                np.savetxt(self.pB_fnames[i],self.pB_data[i])
        

        else:  # load
            listsA = self.load_sols(self.pA_fnames,symName='pA',varnum=2)
            self.pA_data,self.pA_imp,self.pA_callable = listsA
            
            listsB = self.load_sols(self.pB_fnames,symName='pB',varnum=2)
            self.pB_data,self.pB_imp,self.pA_callable = listsB
        
        ta = self.thA
        tb = self.thB
        rulepA = {sym.Indexed('pA',i):self.pA_imp[i](ta,tb) for i in range(self.miter)}
        rulepB = {sym.Indexed('pB',i):self.pB_imp[i](tb,ta) for i in range(self.miter)}
        
        self.rule_p_AB = {**rulepA,**rulepB}
        
    def generate_p(self,k):
        ta = self.thA
        tb = self.thB
        
        if k == 0:
            # g0 is 0
            self.pA_data.append(np.zeros((self.NB,self.NA)))
            self.pB_data.append(np.zeros((self.NA,self.NB)))
            
            self.pA_imp.append(implemented_function('pA0', lambda x: 0))
            self.pB_imp.append(implemented_function('pB0', lambda x: 0))
            
            self.pA_callable.append(0)
            self.pB_callable.append(0)
            
            return
        
        # put these implemented functions into the expansion
        ruleA = {sym.Indexed('pA',i):self.pA_imp[i](ta,tb) for i in range(k)}
        ruleB = {sym.Indexed('pB',i):self.pB_imp[i](tb,ta) for i in range(k)}
        
        rule = {**ruleA, **ruleB}
        
        ph_impA = self.ph_symA[k].subs(rule)
        ph_impB = self.ph_symB[k].subs(rule)
        
        lam_hetA = lambdify([ta,tb],ph_impA)
        lam_hetB = lambdify([tb,ta],ph_impB)
        
        pA_data = np.zeros((self.NB,self.NA,len(self.interval)))
        pB_data = np.zeros((self.NA,self.NB,len(self.interval)))
        
        # no choice but to double loop because of interp2d.
        for i in range(self.NA):
            #if i % 10 == 1:
            #    print(k,i/self.N)
                
            for j in range(self.NB):
                s = self.interval
                a, b = self.A_array[i], self.B_array[j]
                
                intA = np.exp(self.kappa*s)*lam_hetA(a-s,b-s)
                intB = np.exp(self.kappa*s)*lam_hetB(b-s,a-s)
                
                pA_data[j,i,:] = intA
                pB_data[i,j,:] = intB
        
        # integrate
        pA_data = np.sum(pA_data,axis=-1)*self.ds
        pB_data = np.sum(pB_data,axis=-1)*self.ds
        
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.matshow(pA_data,cmap='viridis')
            
            ax.set_ylabel('A')
            ax.set_xlabel('B')
            ax.set_title('pA sum'+str(k))
            
        #print(np.shape(pA_data))
        # turn into interpolated 2d function (inputs automatically taken mod T)
        pA_interp = interp2d(self.A_array,self.B_array,pA_data,bounds_error=False)
        pB_interp = interp2d(self.B_array,self.A_array,pB_data,bounds_error=False)
        
        
        if False:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            
            ax.plot_surface(self.A_mg,self.B_mg,pA_interp(self.A_array,self.B_array),cmap='viridis')
            ax.view_init(azim=0, elev=90)
            ax.set_xlabel('A')
            ax.set_ylabel('B')
            ax.set_title('pA sum'+str(k))
        
        pA_imp = implemented_function('pA_'+str(k),self.myFunMod2A(pA_interp))
        pB_imp = implemented_function('pB_'+str(k),self.myFunMod2B(pB_interp))
        
        #print(pA_imp)
        #self.gx_imp.append(implemented_function('gx_'+str(k),self.myFunMod(fnx)))
        #pA_lam = lambdify([ta,tb],pA_imp)
        #pB_lam = lambdify([tb,ta],pB_imp)
        
        self.pA_data.append(pA_data)
        self.pB_data.append(pB_data)
        
        self.pA_imp.append(pA_imp)
        self.pB_imp.append(pB_imp)
        
        self.pA_callable.append(pA_interp)
        self.pB_callable.append(pB_interp)
        
    def load_h(self):
        
        #self.i_data, self.ix_imp, self.iy_imp = ([] for i in range(3))
        #self.ix_callable, self.iy_callable = ([] for i in range(2))
        self.h_odd_data = []
        self.h_lams = []
        
        if self.recompute_h_odd or not(lib.files_exist(self.h_odd_fnames)):
            ta = self.thA
            tb = self.thB
            
            #self.pA = Sum(eps**i_sym*Indexed('pA',i_sym),(i_sym,1,max_idx)).doit()
            rule1 = {Indexed('zx',i):Indexed('zxA',i) for i in range(self.miter)}
            rule2 = {Indexed('zy',i):Indexed('zyA',i) for i in range(self.miter)}
            
            z_rule = {**rule1,**rule2,**{'psi':self.pA}}
            z = self.z_expansion.subs(z_rule)
            
            h_collected = sym.collect(sym.expand(self.cA.dot(z)),self.eps)
            
            #print(h_collected)
            
            rule = {**self.rule_p_AB,
                    **self.rule_g_AB,
                    **self.rule_z_AB,
                    **self.rule_LC_AB}
            
            
            for i in range(self.miter):
                
                collected = h_collected.coeff(self.eps,i)
                collected_and_sub = collected.subs(rule)
                print(rule)
                print(collected)
                print(collected_and_sub)
                print(i,collected_and_sub.subs({'thA':'t','thB':'t+eta'}))
                print()
                h_lam = sym.lambdify([ta,tb],collected_and_sub)
                self.h_lams.append(h_lam)
                

            for k in range(self.miter):
                print(k)
                data = self.generate_h_odd(k)
                self.h_odd_data.append(data)
                np.savetxt(self.h_odd_fnames[k],data)
        else:
            for k in range(self.miter):
                self.h_odd_data.append(np.loadtxt(self.h_odd_fnames[k]))
        
    def generate_h_odd(self,k):
        """
        interaction functions
        
        note to self: see nb page 130 for notes on indexing in sums.
        need to sum over to index N-1 out of size N to avoid
        double counting boundaries in mod operator.
        """
        
        
        
        #p_mg = np.zeros((self.N-1,self.N-1))
        h_mg = np.zeros((self.NB,self.NA))
        
        for i in range(self.NA):
            for j in range(self.NB):
                t = self.A_array[i]
                eta = self.B_array[j]
                
                h_mg[j,i] = self.h_lams[k](t,t+eta)
                
        #print(np.shape(h_mg))
        #print(np.shape(self.A_mg[:-1,:-1]))
        
        if False:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            #ax.plot_surface(self.A_mg,self.B_mg,
            #                self.pA_callable[k](self.A_array,self.B_array),
            #                cmap='viridis')
            
            ax.plot_surface(self.A_mg,self.B_mg,
                            h_mg,
                            cmap='viridis')
            ax.set_xlabel('t')
            ax.set_ylabel('t+eta')
            ax.set_title('h sum'+str(k))
        
        
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.matshow(h_mg,cmap='viridis')
            
            ax.set_ylabel('t')
            ax.set_xlabel('t+eta')
            ax.set_title('h sum'+str(k))
            
        # sum along axis to get final form
        h = np.sum(h_mg,axis=1)/(self.NA)
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
        
    def integrand(self,s,fn,inA,inB):
        
        return np.exp(self.kappa*s)*self.bispeu(fn,inA-s,inB-s)
    
    def load_hfuns(self):
        pass
        # get autocorrelation of G with Z
        #f1 = Matrix([[self.zx_callable[0](A_array)],
        #             [self.zy_callable[0](A_array)]])
        
        #f2 = self.CGL_coupling(x1,y1,x2,y2)
    
        #print(f1)
        #print(self.cA)

    def load_monodromy(self):
        """
        if monodromy data exists, load. if DNE or recompute required, compute here.
        """
        if self.recompute_monodromy or not(lib.files_exist([self.monodromy_fname])):
            init = np.reshape(self.eye,4)
            
            sol = solve_ivp(lib.monodromy,[0,self.tLC[-1]],init,
                            args=(self.jacLC,),
                            method='RK45',dense_output=True)
            
            self.sol = sol.sol(self.tLC).T
            
            if False:
            
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(self.tLC,self.sol[:,0])
                ax.plot(self.tLC,self.sol[:,1])
                ax.plot(self.tLC,self.sol[:,2])
                ax.plot(self.tLC,self.sol[:,3])
                ax.set_title('Monodromy sol')
                
                plt.show(block=True)
            
            self.M = np.reshape(self.sol[-1,:],(2,2))
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
        
        
    def load_gh(self):
        # load het. functions h if they exist. otherwise generate.
        
        if self.recompute_gh or not(lib.files_exist(self.ghx_fnames,self.ghy_fnames)):
            self.generate_gh()  # populate ghx_list, ghy_list

            for i in range(self.miter):
                dill.dump(self.ghx_imp[i],open(self.ghx_fnames[i],'wb'),recurse=True)
                dill.dump(self.ghy_imp[i],open(self.ghy_fnames[i],'wb'),recurse=True)

        else:
            self.ghx_imp = lib.load_dill(self.ghx_fnames)
            self.ghy_imp = lib.load_dill(self.ghy_fnames)
            
    def generate_gh(self):
        """
        generate heterogeneous terms for the Floquet eigenfunctions g.

        Returns
        -------
        list of symbolic heterogeneous terms in self.ghx_imp, self.ghy_imp.

        """
        
        self.ghx_imp, self.ghy_imp = ([] for i in range(2))
        
        # get the general expression for h before plugging in g.
        self.hx = 0
        self.hy = 0
        
        for i in range(2,self.trunc_derivative+1):
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
        
        # collect all psi terms into list of some kind
        self.tempx = sym.collect(self.hx[0],self.psi,evaluate=False)
        self.tempy = sym.collect(self.hy[0],self.psi,evaluate=False)
        
        counter = 0
        while (counter <= self.miter):
            # save current term
            self.ghx_imp.append(self.tempx[self.psi**counter])
            self.ghy_imp.append(self.tempy[self.psi**counter])
        
            counter += 1
            
        # substitute limit cycle. maybe move elsewhere.
        for i in range(len(self.ghx_imp)):
            self.ghx_imp[i] = self.ghx_imp[i].subs({self.x:self.LC_x,
                                                    self.y:self.LC_y,
                                                    sym.Indexed('gx',0):s(0),
                                                    sym.Indexed('gy',0):s(0)})
            self.ghy_imp[i] = self.ghy_imp[i].subs({self.x:self.LC_x,
                                                    self.y:self.LC_y,
                                                    sym.Indexed('gx',0):s(0),
                                                    sym.Indexed('gy',0):s(0)})
            
    def load_g(self):
        
        # load all g or recompute or compute new.
        self.g_data, self.gx_imp, self.gy_imp = ([] for i in range(3))
        self.gx_callable, self.gy_callable = ([] for i in range(2))

        if self.recompute_g or not(lib.files_exist(self.g_fnames)):  # generate
            
            print('* Computing...',end=' ')
            
            for i in range(self.miter):
                print('g_'+str(i),end=', ')
                data = self.generate_g(i)
                self.g_data.append(data)
                np.savetxt(self.g_fnames[i],data)
            print()
            
        else:   # load
            
            self.g_data,self.gx_imp,self.gy_imp = self.load_sols(self.g_fnames,symName='g')
            
        rulex = {sym.Indexed('gx',i):self.gx_imp[i](self.t) for i in range(len(self.g_fnames))}
        ruley = {sym.Indexed('gy',i):self.gy_imp[i](self.t) for i in range(len(self.g_fnames))}
        self.rule_g = {**rulex,**ruley}
        
        
        for i in range(len(self.g_data)):
            self.gx_callable.append(lambdify(self.t,self.gx_imp[i](self.t)))
            self.gy_callable.append(lambdify(self.t,self.gy_imp[i](self.t)))
        
        # coupling
        thA = self.thA
        thB = self.thB
        
        rule_gxA = {Indexed('gxA',i):self.gx_imp[i](thA) for i in range(self.miter)}
        rule_gyA = {Indexed('gyA',i):self.gy_imp[i](thA) for i in range(self.miter)}
        
        rule_gxB = {Indexed('gxB',i):self.gx_imp[i](thB) for i in range(self.miter)}
        rule_gyB = {Indexed('gyB',i):self.gy_imp[i](thB) for i in range(self.miter)}
        
        self.rule_g_AB = {**rule_gxA,**rule_gyA,**rule_gxB,**rule_gyB}

    def generate_g(self,k,max_iter=200,rel_tol=1e-12):
        # load kth expansion of g for k >= 0
        
        if k == 0:
            # g0 is 0
            
            #self.g_data.append()
            self.gx_imp.append(implemented_function('gx_0', lambda t: 0))
            self.gy_imp.append(implemented_function('gy_0', lambda t: 0))
            
            return np.zeros((self.TN,2))
        
        rulex = {sym.Indexed('gx',i):self.gx_imp[i](self.t) for i in range(k)}
        ruley = {sym.Indexed('gy',i):self.gy_imp[i](self.t) for i in range(k)}
        rule = {**rulex,**ruley}
        
        # apply replacement 
        self.ghx_imp[k] = self.ghx_imp[k].subs(rule)
        self.ghy_imp[k] = self.ghy_imp[k].subs(rule)
        
        # lambdify heterogeneous terms for use in integration
        hetx_lam = lambdify(self.t,self.ghx_imp[k])
        hety_lam = lambdify(self.t,self.ghy_imp[k])
        
        # find intial condtion
        if k == 1:
            # pick appropriately normalized initial condition.
            # see pg 5, wilson 2020.

            #init = -self.g1_init*2*np.pi
            init = self.g1_init
            max_iter = 1
            
        else:
            init = [0,0]
            
            # Newton
            rel_err = 10
            counter = 0
            
            while (rel_err > rel_tol) and (counter < max_iter):
                dx, sol = lib.get_newton_jac(self.dg,-self.tLC,init,hetx_lam,hety_lam,k,eps=1e-1,
                                             return_sol=True)
                
                rel_err = np.amax(np.abs(sol[-1,:]-sol[0,:]))/np.amax(np.abs(sol))
                init += dx
                counter += 1


                if False:
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(self.tLC,sol[:,0])
                    ax.plot(self.tLC,sol[:,1])
                    ax.set_title('g sol_unpert'+str(k))
                    plt.show(block=True)
                    print(counter,rel_err,init)
                
                #print(counter,np.amax(np.abs(dx)),dx,rel_tol,k)
                if counter == max_iter-1:
                    print('WARNING: max iter reached in newton call')
                    
        # get full solution
        
        #if k == 3:
        #    #init = [-3.174675640774596852e-01,6.828310976413081157e-01]
        #    init = [0.4275914,-0.7147441]

        sol = solve_ivp(self.dg,[0,-self.tLC[-1]],
                        init,args=(hetx_lam,hety_lam,k),
                        method='RK45',dense_output=True,
                        rtol=1e-4,atol=1e-10)
            
        gu = sol.sol(-self.tLC).T[::-1,:]

        if False:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.tLC,gu[:,0])
            ax.plot(self.tLC,gu[:,1])
            ax.set_title('gu'+str(k))
            print('final init',init)
            plt.show(block=True)
            
            #time.sleep(60)

        # save soluton as lambda functions
        fnx = interp1d(self.tLC,gu[:,0],fill_value='extrapolate')
        fny = interp1d(self.tLC,gu[:,1],fill_value='extrapolate')
        
        self.gx_imp.append(implemented_function('gx_'+str(k),self.myFunMod(fnx)))
        self.gy_imp.append(implemented_function('gy_'+str(k),self.myFunMod(fny)))
        
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
                fnx = interp1d(self.tLC,data[:,0],fill_value='extrapolate')
                fny = interp1d(self.tLC,data[:,1],fill_value='extrapolate')
                
                xtemp = implemented_function(symName+'x_'+str(i), self.myFunMod(fnx))
                ytemp = implemented_function(symName+'y_'+str(i), self.myFunMod(fny))
            else:
                #print('loaded pa', np.shape(data))
                #print(np.shape(self.A_array),np.shape(self.B_array))
                if symName == 'pA':
                    fnx = interp2d(self.A_array,self.B_array,data,bounds_error=False)
                if symName == 'pB':
                    fnx = interp2d(self.B_array,self.A_array,data,bounds_error=False)
                    
                xtemp = implemented_function(symName+'_'+str(i),self.myFunMod2A(fnx))
                ytemp = fnx
                
                
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
            for i in range(self.miter):
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
        
        for j in range(1,self.trunc_derivative+1):
            p1 = lib.kProd(j,self.dx_vec)
            p2 = kp(p1,sym.eye(2))
        
            d1 = lib.vec(lib.df(self.CGL_sym[0],self.x_vec,j+1))
            d2 = lib.vec(lib.df(self.CGL_sym[1],self.x_vec,j+1))
            
            self.ax += (1/math.factorial(j)) * p2*d1
            self.ay += (1/math.factorial(j)) * p2*d2
        
        self.A = sym.zeros(2,2)
        
        self.A[:,0] = self.ax
        self.A[:,1] = self.ay
        
        
        
        het = self.A*self.z_expansion
        
        
        # expand all terms
        self.hetx = sym.expand(het[0].subs([(self.dx,self.gx),(self.dy,self.gy)]))
        self.hety = sym.expand(het[1].subs([(self.dx,self.gx),(self.dy,self.gy)]))
        
        # collect all psi terms into factors of pis^k
        self.hetx_powers = sym.collect(self.hetx,self.psi,evaluate=False)
        self.hety_powers = sym.collect(self.hety,self.psi,evaluate=False)
    
    
        self.hetx_list = []
        self.hety_list = []
        
        counter = 0
        while (counter <= self.miter):
            
            # save current term
            self.hetx_list.append(self.hetx_powers[self.psi**counter])
            self.hety_list.append(self.hety_powers[self.psi**counter])
        
            counter += 1
        
        # substitute limit cycle
        for i in range(len(self.ghx_imp)):
            self.hetx_list[i] = self.hetx_list[i].subs({self.x:self.LC_x,
                                                        self.y:self.LC_y,
                                                        sym.Indexed('gx',0):s(0),
                                                        sym.Indexed('gy',0):s(0)})
            self.hety_list[i] = self.hety_list[i].subs({self.x:self.LC_x,
                                                        self.y:self.LC_y,
                                                        sym.Indexed('gx',0):s(0),
                                                        sym.Indexed('gy',0):s(0)})
        
            
    def load_z(self):
        
        # load all g or recompute or compute new.
        self.z_data, self.zx_imp, self.zy_imp = ([] for i in range(3))
        self.zx_callable, self.zy_callable = ([] for i in range(2))

        if self.recompute_z or not(lib.files_exist(self.z_fnames)):
            
            print('* Computing...',end=' ')
            for i in range(self.miter):
                print('z_'+str(i), end=', ')
                data = self.generate_z(i)
                self.z_data.append(data)
                np.savetxt(self.z_fnames[i],data)
            print()
            
        else:
            self.z_data, self.zx_imp, self.zy_imp = self.load_sols(self.z_fnames,symName='z')
            
        for i in range(len(self.z_data)):
            
            self.zx_callable.append(lambdify(self.t,self.zx_imp[i](self.t)))
            self.zy_callable.append(lambdify(self.t,self.zy_imp[i](self.t)))
        
        # coupling
        thA = self.thA
        thB = self.thB
        
        rule_zxA = {Indexed('zxA',i):self.zx_imp[i](thA) for i in range(self.miter)}
        rule_zyA = {Indexed('zyA',i):self.zy_imp[i](thA) for i in range(self.miter)}
        
        rule_zxB = {Indexed('zxB',i):self.zx_imp[i](thB) for i in range(self.miter)}
        rule_zyB = {Indexed('zyB',i):self.zy_imp[i](thB) for i in range(self.miter)}
        
        self.rule_z_AB = {**rule_zxA,**rule_zyA,**rule_zxB,**rule_zyB}


    def generate_z(self,k,total_iter=5,rel_tol=1e-4):
        
        # load kth expansion of g for k >= 1
        rulex = {sym.Indexed('zx',i):self.zx_imp[i](self.t) for i in range(k)}
        ruley = {sym.Indexed('zy',i):self.zy_imp[i](self.t) for i in range(k)}
        rule = {**rulex,**ruley,**self.rule_g}
        
        zhx = self.hetx_list[k].subs(rule)
        zhy = self.hety_list[k].subs(rule)
        
        hetx_lam = lambdify(self.t,zhx)
        hety_lam = lambdify(self.t,zhy)
        
        if k == 0:
            init = copy.deepcopy(self.z0_init)
            max_iter = 1
        else:
            init = [0,0]
        
            # Newton
            max_iter = 200
            rel_err = 10
            
            counter = 0
            while (rel_err > rel_tol) and (counter < max_iter):
                
                dx,sol = lib.get_newton_jac(self.dz,-self.tLC,init,hetx_lam,hety_lam,k,eps=1e-1,
                                            return_sol=True)
                
                rel_err = np.amax(np.abs(sol[-1,:]-sol[0,:]))/np.amax(np.abs(sol))
                init += dx
                counter += 1
                
                if counter == max_iter-1:
                    print('WARNING: max iter reached in newton call')
            
                if False:
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(self.tLC,sol[:,0])
                    ax.plot(self.tLC,sol[:,1])
                    ax.set_title('z sol_unpert'+str(k))
                    plt.show(block=True)
                    print(counter,rel_err)
                    
                    
        sol = solve_ivp(self.dz,[0,-self.tLC[-1]],init,args=(hetx_lam,hety_lam,k),
                        method='RK45',dense_output=True,
                        rtol=1e-5,atol=1e-10)
            
        zu = sol.sol(-self.tLC).T[::-1,:]
        
        
        if False:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.tLC,zu[:,0])
            ax.plot(self.tLC,zu[:,1])
            ax.set_title('zu'+str(k))
            print('final init',init)
            plt.show(block=True)
        
        if k == 0:
            # normalize
            dLC = lib.rhs([np.cos(0),np.sin(0)],0,self.f)
            zu = zu/(np.dot(dLC,zu[0,:]))
        
        fnx = interp1d(self.tLC,zu[:,0],fill_value='extrapolate')
        fny = interp1d(self.tLC,zu[:,1],fill_value='extrapolate')
        
        self.zx_imp.append(implemented_function('zx_'+str(k), self.myFunMod(fnx)))
        self.zy_imp.append(implemented_function('zy_'+str(k), self.myFunMod(fny)))
        
        self.zx_callable.append(lambdify(self.t,self.zx_imp[k](self.t)))
        self.zy_callable.append(lambdify(self.t,self.zy_imp[k](self.t)))
        
        return zu
    

    def load_i(self):
        
        # load all g or recompute or compute new.
        self.i_data, self.ix_imp, self.iy_imp = ([] for i in range(3))
        self.ix_callable, self.iy_callable = ([] for i in range(2))

        if self.recompute_i or not(lib.files_exist(self.i_fnames)):
            
            print('* Computing...',end=' ')
            for i in range(self.miter):
                print('i_'+str(i), end=', ')
                data = self.generate_i(i)
                self.i_data.append(data)
                np.savetxt(self.i_fnames[i],data)
            print()
            
        else:
            self.i_data, self.ix_imp, self.iy_imp = self.load_sols(self.i_fnames,symName='i')
            
        
        for i in range(len(self.i_data)):
            self.ix_callable.append(lambdify(self.t,self.ix_imp[i](self.t)))
            self.iy_callable.append(lambdify(self.t,self.iy_imp[i](self.t)))
    
    def generate_i(self,k,total_iter=5,rel_tol=1e-3):
        
        # load kth expansion of g for k >= 1
        rulex = {sym.Indexed('zx',i):self.ix_imp[i](self.t) for i in range(k)}
        ruley = {sym.Indexed('zy',i):self.iy_imp[i](self.t) for i in range(k)}
        rule = {**rulex,**ruley,**self.rule_g}

        ihx = self.hetx_list[k].subs(rule)
        ihy = self.hety_list[k].subs(rule)
        
        hetx_lam = lambdify(self.t,ihx)
        hety_lam = lambdify(self.t,ihy)
        
        
        if k == 0:
            #init = -copy.deepcopy(self.i0_init)/(2*np.pi)
            init = copy.deepcopy(self.i0_init)  # /(2*np.pi)
            max_iter = 0
            
        else:
            init = [0,0]
        
            # Newton
            
            max_iter = 200
            
            rel_err = 10
            
            counter = 0
            while (rel_err > rel_tol) and (counter < max_iter):
                
                dx,sol = lib.get_newton_jac(self.di,-self.tLC,init,hetx_lam,hety_lam,k,eps=1e-1,
                                            return_sol=True)
                
                rel_err = np.amax(np.abs(sol[-1,:]-sol[0,:]))/np.amax(np.abs(sol))
                
                init += dx
                counter += 1
                
                if counter == max_iter-1:
                    print('WARNING: max iter reached in newton call')
            
            
        
        sol = solve_ivp(self.di,[0,-self.tLC[-1]],init,
                        args=(hetx_lam,hety_lam,k),
                        method='RK45',dense_output=True)
            
        iu = sol.sol(-self.tLC).T[::-1,:]
        
        
        if k == 1:  # normalize
            
            gx = lambdify(self.t,self.gx_imp[1](self.t))
            gy = lambdify(self.t,self.gy_imp[1](self.t))
            
            zx = lambdify(self.t,self.zx_imp[0](self.t))
            zy = lambdify(self.t,self.zy_imp[0](self.t))
            
            ix = lambdify(self.t,self.ix_imp[0](self.t))
            iy = lambdify(self.t,self.iy_imp[0](self.t))
            
            F = lib.rhs([np.cos(0),np.sin(0)],0,self.f)
            g1 = np.array([gx(0),gy(0)])
            z0 = np.array([zx(0),zy(0)])
            i0 = np.array([ix(0),iy(0)])
            
            J = self.jacLC(0)
            i1 = iu[0,:]
            
            ijg = np.dot(i0,np.dot(J,g1))
            be = (self.kappa - ijg - np.dot(i1,F))/(np.dot(z0,F))
            
            init = iu[0,:] + be*z0
            
            
            sol = solve_ivp(self.di,[0,self.tLC[-1]],init,
                            args=(hetx_lam,hety_lam,k),
                            method='RK45',dense_output=True)
            
            iu = sol.sol(self.tLC).T
        
    
        fnx = interp1d(self.tLC,iu[:,0],fill_value='extrapolate')
        fny = interp1d(self.tLC,iu[:,1],fill_value='extrapolate')
        
        self.ix_imp.append(implemented_function('ix_'+str(k), self.myFunMod(fnx)))
        self.iy_imp.append(implemented_function('iy_'+str(k), self.myFunMod(fny)))
        
        self.ix_callable.append(lambdify(self.t,self.ix_imp[k](self.t)))
        self.iy_callable.append(lambdify(self.t,self.iy_imp[k](self.t)))
        
        return iu


    def dg(self,t,z,hetx,hety,order):
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
        het = 0.5*np.array([hetx(t),hety(t)])
        
        return hom + het
    
    def dz(self,t,z,hetx,hety,order):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = -np.dot(self.jacLC(t).T+order*self.kappa*self.eye,z)
        het = -np.array([hetx(t),hety(t)])
        
        out = hom + het
        
        return out
    
    def di(self,t,z,hetx,hety,order):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        #hom = -np.dot(self.jacLC(t).T+self.kappa*(order-1)*self.eye,z)
        #het = -np.array([hetx(t),hety(t)])
        
        hom = -np.dot(self.jacLC(t).T+self.kappa*(order-1)*self.eye,z)
        het = -np.array([hetx(t),hety(t)])
        
        
        out = hom + het
        
        return out
    
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
        
        return lambda xA=self.thA,xB=self.thB: fn(np.mod(xA,self.T),np.mod(xB,self.T))
    
    
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
    a = CGL(recompute_gh=False,
            recompute_g=False,
            recompute_het=False,
            recompute_z=False,
            recompute_i=False,
            recompute_ph=False,
            recompute_p=False,
            recompute_h_odd=False,
            trunc_order=3,
            trunc_derviative=2,
            q=2*np.pi)
    
    #for i in range(a.miter):
    #    lib.plot(a,'g'+str(i))
        
    for i in range(a.miter):
        lib.plot(a,'z'+str(i))
        
    for i in range(a.miter):
        lib.plot(a,'i'+str(i))
        
    for i in range(a.miter):
        lib.plot(a,'hodd'+str(i))
        
        
    for i in range(a.miter):
        lib.plot(a,'pA'+str(i))
    plt.show(block=True)
    
    lib.plot(a,'surface_z')
    lib.plot(a,'surface_i')
    
    
    # check total hodd
    ve = .5
    h = 0
    for i in range(4):
        h += ve**(i+1)*a.h_odd_data[i]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(h)
    
    plt.show(block=True)
    
    
    
if __name__ == "__main__":
    
    import cProfile
    import re
    cProfile.runctx('main()',globals(),locals(),'profile.pstats')
    #cProfile.runctx('main()',globals(),locals())

    #main()
