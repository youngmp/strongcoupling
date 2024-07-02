"""
StrongCoupling.py computes the higher-order interaction functions from
Park and Wilson 2020 for $N=2$ models and one Floquet multiplier.
In broad strokes, this library computes functions in the following order:

* Use the equation for $\Delta x$ (15) to produce a hierarchy of
ODEs for $g^{(k)}$ and solve. (Wilson 2020)
* Do the same using (30) and (40) to generate a hierarchy of ODEs
for $Z^{(k)}$ and $I^{(k)}$, respectively. (Wilson 2020)
* Solve for $\phi$ in terms of $\\theta_i$, (13), (14) (Park and Wilson 2020)
* Compute the higher-order interaction functions (15) (Park and Wilson 2020)



Notes:
- ``pA`` requires endpoint=False. make sure corresponding `dxA`s are used.

"""
import copy
import lib.lib_sym as slib
import lib.lib_sym2 as slib2

#import lib.lib as lib
from lib import lib

from lib.interp_basic import interp_basic as interpb
from lib.interp2d_basic import interp2d_basic as interp2db
#from lam_vec import lam_vec

from lib import fnames

#import inspect
import time
import os

import math
#import sys
#import multiprocessing as multip
import tqdm
#from pathos.pools import ProcessPool
from pathos.pools import _ProcessPool

import scipy.interpolate as si
import numpy as np
#import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt

import dill

from sympy import Matrix, symbols, Sum, Indexed, collect, expand
from sympy import sympify as s
from sympy.physics.quantum import TensorProduct as kp
from sympy.utilities.lambdify import lambdify, implemented_function

#import pdoc

imp_fn = implemented_function

#from interpolate import interp1d
#from scipy.interpolate import interp1d#, interp2d
from scipy.interpolate import interp2d
from scipy.integrate import solve_ivp


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


# updated coupling class. much more efficient
# less user unfriendly.
class StrongCoupling2(object):

    def __init__(self,system1,system2,
                 NH=100,trunc_deriv=3,
                 
                 _n=(None,1),_m=(None,1),
                 
                 method='LSODA', # good shit                 
                 max_n=-1, # Fourier terms. -1 = no truncation
                 
                 save_fig=False,                 
                 recompute_list=[]):

        
        self.save_fig = save_fig

        self._expand_kws = {'basic':True,'deep':True,'power_base':False,
                            'power_exp':False,'mul':True,'log':False,
                            'multinomial':True}

        system1 = copy.deepcopy(system1)
        system2 = copy.deepcopy(system2)

        assert(type(_n) is tuple);assert(type(_m) is tuple)
        #assert(_n[0] in system1.pardict);assert(_m[0] in system2.pardict)

        self.system1 = system1;self.system2 = system2
        self.recompute_list = recompute_list

        self.trunc_deriv = trunc_deriv

        self.om = 1

        self._n = (1,1)
        self._m = (1,1)

        ths_str = ' '.join(['th'+str(i) for i in range(2)])
        self.ths = symbols(ths_str,real=True)

        # discretization for p_X,p_Y
        self.NP = NH
        self.x,self.dx = np.linspace(0,2*np.pi,self.NP,retstep=True,endpoint=False)

        self.an,self.dan = np.linspace(0,2*np.pi*self._n[1],self.NP*self._n[1],
                                       retstep=True,endpoint=False)

        # discretization for H_X,H_Y
        self.NH = NH
        self.bn,self.dbn = np.linspace(0,2*np.pi*self._m[1],NH,
                                       retstep=True,endpoint=False)

        self.be,self.dbe = np.linspace(0,2*np.pi*self._m[1],NH*self._m[1],
                                       retstep=True,endpoint=True)

        self.bne,self.dbne = np.linspace(0,2*np.pi*self._n[1],NH*self._n[1],
                                         retstep=True,endpoint=True)        


        system1.h['dat']={};system1.h['imp']={};system1.h['lam']={}
        system1.h['sym']={};system1.p['dat']={};system1.p['imp']={}
        system1.p['lam']={};system1.p['sym']={}

        # global replacement rules based on system index
        system1.rule_lc={};system1.rule_g={};system1.rule_z={}
        system1.rule_i={};system1.rule_p={}

        if not(system1.forcing):
            system2.rule_lc={};system2.rule_g={};system2.rule_z={}
            system2.rule_i={};system2.rule_p={}

        for j,key in enumerate(system1.var_names):
            in1 = self.ths[system1.idx]
            system1.rule_lc[system1.syms[j]] = system1.lc['imp_'+key](in1)

        for key in system1.var_names:
            for k in range(system1.miter):
                in1 = self.ths[system1.idx]

                kg = sym.Indexed('g_'+system1.model_name+'_'+key,k)
                system1.rule_g[kg] = system1.g['imp_'+key][k](in1)

                kz = sym.Indexed('z_'+system1.model_name+'_'+key,k)
                system1.rule_z[kz] = system1.z['imp_'+key][k](in1)

                ki = sym.Indexed('i_'+system1.model_name+'_'+key,k)
                system1.rule_i[ki] = system1.i['imp_'+key][k](in1)

        self._init_noforce(system1,system2)


    def _init_noforce(self,system1,system2):
        self.t,self.dt = np.linspace(0,2*np.pi,100,retstep=True)

        pfactor1 = int((np.log(.05)/system1.kappa_val))
        pfactor2 = int((np.log(.05)/system2.kappa_val))
        
        self.pfactor = max([pfactor1,pfactor2])
        
        # just keep these here.
        system2.h['dat']={};system2.h['imp']={};system2.h['lam']={}
        system2.h['sym']={};system2.p['dat']={};system2.p['imp']={}
        system2.p['lam']={};system2.p['sym']={}
        
        
        for j,key in enumerate(system2.var_names):
            in1 = self.ths[system2.idx]
            system2.rule_lc[system2.syms[j]] = system2.lc['imp_'+key](in1)
        
        for key in system2.var_names:
            for k in range(system2.miter):
                in1 = self.ths[system2.idx]
                
                kg = sym.Indexed('g_'+system2.model_name+'_'+key,k)
                system2.rule_g[kg] = system2.g['imp_'+key][k](in1)

                kz = sym.Indexed('z_'+system2.model_name+'_'+key,k)
                system2.rule_z[kz] = system2.z['imp_'+key][k](in1)
                
                ki = sym.Indexed('i_'+system2.model_name+'_'+key,k)
                system2.rule_i[ki] = system2.i['imp_'+key][k](in1)
        
        
        fnames.load_fnames_nm(system1,self)
        fnames.load_fnames_nm(system2,self)
        slib2.load_coupling_expansions(system1)
        slib2.load_coupling_expansions(system2)

        # mix up terms. G and K
        self.load_k_sym(system1,system2)
        self.load_k_sym(system2,system1)

        # load odes for p_i
        self.load_p_sym(system1)
        self.load_p_sym(system2)

        for i in range(system1.miter):
            self.load_p(system1,system2,i)
            self.load_p(system2,system1,i)

        self.load_h_sym(system1)
        self.load_h_sym(system2)

        for i in range(system1.miter):
            self.load_h(system1,system2,i)
            self.load_h(system2,system1,i)

    def load_k_sym(self,system1,system2):
        
        """
        G is the full coupling function expanded.
        K is the same, but indexed by powers of eps.

        """
            
        # check that files exist
        val = not(os.path.isfile(system1.G['fname']))
        val += not(os.path.isfile(system1.K['fname']))

        files_dne = val

        system1.G['sym'] = {}
        system1.K['sym'] = {}
        if 'k_'+system1.model_name in self.recompute_list or files_dne:

            rule_trunc = {}
            for k in range(system1.miter,system1.miter+500):
                rule_trunc.update({system1.eps**k:0})

            self.generate_k_sym(system1,system2)

            # now collect terms
            # G contains full summation
            # K contains terms of summation indexed by power
            
            for key in system1.var_names:

                tmp = expand(system1.G['sym'][key],**self._expand_kws)
                system1.G['sym'][key] = tmp.subs(rule_trunc)
                
                system1.K['sym'][key] = []
                for k in range(system1.miter):
                    e_term = tmp.coeff(system1.eps,k)
                    system1.K['sym'][key].append(e_term)

            dill.dump(system1.G['sym'],open(system1.G['fname'],'wb'),
                      recurse=True)
            dill.dump(system1.K['sym'],open(system1.K['fname'],'wb'),
                      recurse=True)
                
        else:
            system1.G['sym'] = dill.load(open(system1.G['fname'],'rb'))
            system1.K['sym'] = dill.load(open(system1.K['fname'],'rb'))

        system1.G['vec'] = sym.zeros(system1.dim,1)

        

        for i,key in enumerate(system1.var_names):
            system1.G['vec'][i] = [system1.G['sym'][key]]

    def generate_k_sym(self,system1,system2):
        """
        generate terms involving the coupling term (see K in paper).
        Equations (8), (9) in Park, Wilson 2021
        """

        psym = system1.pardict_sym # shorten for readability
        fn = system1.coupling # coupling function

        # all variables and d_variables
        var_names_all = system1.var_names + system2.var_names
        z = sym.Matrix(system1.syms + system2.syms)
        dz = sym.Matrix(system1.dsyms + system2.dsyms)

        # get full coupling term for given oscillator
        system1.G['sym_fn'] = sym.flatten(fn(z,psym,option='sym',
                                             idx=system1.idx))

            
        
        # 0 and 1st derivative
        c_temp = {}
        for key_idx,key in enumerate(system1.var_names):
            c_temp[key] = system1.G['sym_fn'][key_idx]
            
            d = lib.df(system1.G['sym_fn'][key_idx],z,1).dot(dz)
            c_temp[key] += d
            
        # 2nd + derivative        
        for key_idx,key in enumerate(system1.var_names):
            for k in range(2,self.trunc_deriv+1):
                kp = lib.kProd(k,dz)
                da = lib.vec(lib.df(system1.G['sym_fn'][key_idx],z,k))
                c_temp[key] += (1/math.factorial(k))*kp.dot(da)

        # save 
        for key in system1.var_names:
            system1.G['sym'][key] = c_temp[key]

                        
        # replace dx with (g expansion in eps).
        rule = {}

        for key_idx,key in enumerate(system1.var_names):
            rule.update({system1.dsyms[key_idx]:system1.g[key+'_eps']})
        for key_idx,key in enumerate(system2.var_names):
            rule.update({system2.dsyms[key_idx]:system2.g[key+'_eps']})

        

        for key in system1.var_names:
            system1.G['sym'][key] = system1.G['sym'][key].subs(rule)
        
    def load_p_sym(self,system1):
        """
        generate/load the het. terms for psi ODEs.
            
        to be solved using integrating factor meothod.        
        system*.p[k] is the forcing function for oscillator * of order k
        """

        file_dne = not(os.path.isfile(system1.p['fname']))

        eps = system1.eps

        if 'p_'+system1.model_name in self.recompute_list or file_dne:
            print('* Computing p symbolic...')
            
            rule_trunc = {}
            for k in range(system1.miter,system1.miter+500):
                rule_trunc.update({eps**k:0})

            v1 = system1.i['vec']; v2 = system1.G['vec']
            
            tmp = eps*v1.dot(v2)
            tmp = expand(tmp,**self._expand_kws)
            
            tmp = tmp.subs(rule_trunc)
            tmp = collect(tmp,system1.eps)
            tmp = collect(expand(tmp),system1.eps)

            for k in range(system1.miter):
                system1.p['sym'][k] = tmp.coeff(system1.eps,k)

            dill.dump(system1.p['sym'],open(system1.p['fname'],'wb'),
                      recurse=True)

        else:
            print('* Loading p symbolic...')
            system1.p['sym'] = dill.load(open(system1.p['fname'],'rb'))

        

    def load_p(self,system1,system2,k):

        fname = system1.p['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))
        
        if 'p_data_'+system1.model_name in self.recompute_list\
           or file_dne:
            
            
            p_data = self.generate_p(system1,system2,k)
                
            np.savetxt(system1.p['fnames_data'][k],p_data)

        else:
            
            p_data = np.loadtxt(fname)

        system1.p['dat'][k] = p_data

        n=self._n[1];m=self._m[1]
        x=self.x;dx=self.dx
        
        p_interp = interp2d([0,0],[x[-1]*n,x[-1]*m],[dx*n,dx*m],p_data,
                            k=9,p=[True,True])

        ta = self.ths[0];tb = self.ths[1]
        name = 'p_'+system1.model_name+'_'+str(k)
        p_imp = imp_fn(name,self.fast_interp_lam(p_interp))
        lamtemp = lambdify(self.ths,p_imp(ta,tb))
        

        if k == 0:
            imp = imp_fn('p_'+system1.model_name+'_0', lambda x: 0*x)
            system1.p['imp'][k] = imp
            system1.p['lam'][k] = 0
            
            
        else:
            system1.p['imp'][k] = p_imp
            system1.p['lam'][k] = lamtemp#p_interp

        if self.save_fig:
            
            fig,axs = plt.subplots()
            axs.imshow(p_data)
            pname = 'figs_temp/p_'
            pname += system1.model_name+str(k)
            pname += '_'+str(n)+str(m)
            pname += '.png'
            plt.savefig(pname)
            plt.close()

        # put these implemented functions into the expansion
        system1.rule_p.update({sym.Indexed('p_'+system1.model_name,k):
                               system1.p['imp'][k](ta,tb)})


    def generate_p(self,system1,system2,k):
        print('p order='+str(k))

        n = self._n[1];m = self._m[1]
        NP = self.NP
        data = np.zeros([n*NP,m*NP])
        
        if k == 0:
            #pA0 is 0 (no forcing function)
            return data

        kappa1 = system1.kappa_val

        rule = {**system1.rule_p,**system2.rule_p,**system1.rule_par,
                **system2.rule_par,**system1.rule_lc,**system2.rule_lc,
                **system1.rule_i,**system2.rule_i,**system1.rule_g,
                **system2.rule_g}
        
        if self.forcing:
            imp0 = imp_fn('forcing',system2)
            lam0 = lambdify(self.ths[1],imp0(self.ths[1]))
            rule.update({system2.syms[0]:imp0(self.ths[1])})

        ph_imp1 = system1.p['sym'][k].subs(rule)

        if ph_imp1 == 0: # keep just in case
            return np.zeros([NP,NP])
        
        lam1 = lambdify(self.ths,ph_imp1)        
        
        x=self.x;dx=self.dx;pfactor=self.pfactor
        sm=np.arange(0,self.T*pfactor,dx)*m
        
        fac = self.om*(1-system1.idx) + system1.idx
        exp1 = exp(fac*sm*system1.kappa_val)

        g_in = np.fft.fft(exp1)
        a_i = np.arange(NP,dtype=int)
        
        #for ll in range(len(an)):
        for ll in range(len(x)):

            f_in = np.fft.fft(lam1(x[ll]*n+self.om*sm,sm))
            conv = np.fft.ifft(f_in*g_in)
            data[(a_i+ll)%int(n*NP),a_i] = conv[-int(m*NP):].real
            #data[ll,:] = conv[-self._m[1]*NP:].real

        return data*self.dan
    
    
    def load_h_sym(self,system):
        """
        also compute h lam
        """

        fname = system.h['fname']
        file_dne = not(os.path.isfile(fname))

        # symbolic h terms
        system.h['sym'] = {}
        
        if 'h_'+system.model_name in self.recompute_list or file_dne:
            print('* Computing H symbolic...')
            
            # simplify expansion for speed
            rule_trunc = {}
            for k in range(system.miter,system.miter+200):
                rule_trunc.update({system.eps**k:0})

            v1 = system.z['vec']; v2 = system.G['vec']
                    
            tmp = v1.dot(v2)
            tmp = sym.expand(tmp,**self._expand_kws)
            
            tmp = tmp.subs(rule_trunc)
            tmp = collect(tmp,system.eps)
            tmp = collect(expand(tmp),system.eps)

            for k in range(system.miter):
                system.h['sym'][k] = tmp.coeff(system.eps,k)

            dill.dump(system.h['sym'],open(fname,'wb'),recurse=True)
                
        else:
            print('* Loading H symbolic...')
            system.h['sym'] = dill.load(open(fname,'rb'))
            
    def load_h(self,system1,system2,k):

        fname = system1.h['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))

        if 'h_data_'+system1.model_name in self.recompute_list\
           or file_dne:
            str1 = '* Computing H {}, order={}...'
            print(str1.format(system1.model_name,k))

            h_data = self.generate_h(system1,system2,k)
            np.savetxt(system1.h['fnames_data'][k],h_data)

        else:
            str1 = '* Loading H {}, order={}...'
            print(str1.format(system1.model_name,k))
            
            h_data = np.loadtxt(fname)

        system1.h['dat'][k] = h_data
        hodd = h_data[::-1] - h_data

        n=self._n[1];m=self._m[1]
        #system1.h['lam'][k] = interpb(self.be,h_data,2*np.pi)
        #system1.h['lam'][k] = interp1d(0,self.x[-1]*n,self.dx*n,
        #                               h_data,p=True,k=9)

        system1.h['lam'][k] = interp1d(self.bn[0],self.bn[-1],self.dbn,
                                       h_data,p=True,k=5)

        if self.save_fig:    
            fig,axs = plt.subplots(figsize=(6,2))

            axs.plot(h_data)

            ptitle = 'h '+system1.model_name
            ptitle += 'o='+str(k)
            ptitle += ', nm='+str(n)+str(m)
            axs.set_title(ptitle)
            
            pname = 'figs_temp/h_'
            pname += system1.model_name+str(k)
            pname += '_'+str(n)+str(m)
            pname += '.png'
            plt.savefig(pname)            
            plt.close()

                
    def generate_h(self,system1,system2,k):

        rule = {**system1.rule_p,**system2.rule_p,**system1.rule_par,
                **system2.rule_par,**system1.rule_lc,**system2.rule_lc,
                **system1.rule_g,**system2.rule_g,**system1.rule_z,
                **system2.rule_z,**system1.rule_i,**system2.rule_i}
        
        if self.forcing:
            imp0 = imp_fn('forcing',system2)
            rule.update({system2.syms[0]:imp0(self.ths[1])})
            
        h_lam = lambdify(self.ths,system1.h['sym'][k].subs(rule))

        """
        x=self.x;dx=self.dx
        n=self._n[1];m=self._m[1]
        
        X,Y = np.meshgrid(x*n,x*m,indexing='ij')
        h_integrand = h_lam(X,Y)
        ft2 = np.fft.fft2(h_integrand)
        # get only those in n:m indices
        # n*i,-m*i
        ft2_new = list([ft2[0,0]])
        ft2_new += list(np.diag(np.flipud(ft2[1:,1:]))[::-1])
        ft2_new = np.asarray(ft2_new)
        ft2_new /= np.sqrt(self.NH)*2

        out = np.fft.ifft(ft2_new).real/(2*np.pi*m)
        print('h mean',np.mean(out))
        return out
        """
        
        bn=self.be;dbn=self.dbe
        #bn=self.be;dbn=self.dbe
        n=self._n[1];m=self._m[1]

        # calculate mean
        #X,Y = np.meshgrid(self.an,self.an*self._m[1],indexing='ij')
        #system1._H0[k] = np.sum(h_lam(X,Y))*self.dan**2/(2*np.pi*self._m[1])**2

        def integrand(x,y):
            return h_lam(y+self.om*x,x)
                
        # calculate coupling function
        h = np.zeros(self.NH)
        for j in range(self.NH):
            #val = np.sum(h_lam(bn[j] + self.om*bn,bn))*dbn
            val = quad(integrand,bn[0],bn[-1],args=(bn[j],),limit=10000)[0]

            h[j] = val

        out = h/(2*np.pi*self._m[1])

        return out
        
    def fast_interp_lam(self,fn):
        """
        interp2db object (from fast_interp)
        """
        return lambda t0=self.ths[0],t1=self.ths[1]: fn(t0,t1)

    
    def save_temp_figure(self,data,k,fn='plot',path_loc='figs_temp/'):
        """
        data should be (TN,dim)
        """

        if (not os.path.exists(path_loc)):
            os.makedirs(path_loc)
        
        fig, axs = plt.subplots(nrows=self.dim,ncols=1)
        
        for j,ax in enumerate(axs):
            key = self.var_names[j]
            ax.plot(self.tLC,data[:,j],label=key)
            ax.legend()
            
        axs[0].set_title(fn+str(k))
        plt.tight_layout()
        plt.savefig(path_loc+fn+str(k)+'.png')
        plt.close()


# below is the legacy strongcoupling class
class StrongCoupling(object):
    
    
    def __init__(self,rhs,coupling,LC_init,var_names,pardict,**kwargs):

        """
        See the defaults dict below for allowed kwargs.
        
        All model parameters must follow the convention
        'parameter_val'. No other underscores should be used.
        the script splits the parameter name at '_' and uses the
        string to the left as the sympy parmeter name.
        
        Reserved names: ...
            
            rhs: callable.
                right-hand side of a model
            coupling: callable.
                coupling function between oscillators
            LC_init: list or numpy array.
                initial condition of limit cycle (must be found manually).
                XPP is useful, otherwise integrate your RHS for various
                initial conditions for long times and extract an initial
                condition close to the limit cycle.
            LC_long_sim_time: float or int.
                Simulation time to compute the trjactory of an initial 
                condition near the limit cycle solution. Used to
                estimate limit cycle initial condition for use 
                in the Newton method. Default: 500
            LC_eps_time: float.
                Approximation of the time error estimate in Newton's
                method, e.g., (difference between initial conditions)/
                (LC_eps_time). Default: 1e-4
            LC_tol: float:
                Error tolerance to stop Newton's method when computing
                the limit cycle. Default: 1e-13
            var_names: list.
                list of variable names as strings
            pardict: dict.
                dictionary of parameter values. dict['par1_val'] = float.
                Make sure to use par_val format, where each parameter name is
                followed by _val.
            recompute_LC: bool.
                If True, recompute limit cycle. If false, load limit cycle if
                limit cycle data exists. Otherwise, compute. Default: False.
            recompute_monodromy: bool.
                If true, recompute kappa, the FLoquet multiplier using the
                monodromy matrix. If false, load kappa if data exists,
                otherwise compute. Default: False.
            recompute_g_sym: bool.
                If true, recompute the symbolic equations for g^k. If false,
                load the symbolic equations if they exist in storage.
                Otherwise, compute. Default: False.
            recompute_g: bool.
                If true, recompute the ODEs for g^k. If false,
                load the data for g^k if they exist in storage.
                Otherwise, compute. Default: False.
            recompute_het_sym: bool.
                If true, recompute the symbolic equations for z^k and i^k.
                If false, load the symbolic equations if they exist in
                storage. Otherwise, compute. Default: False.
            recompute_z: bool.
                If true, recompute the ODEs for z^k. If false,
                load the data for z^k if they exist in storage.
                Otherwise, compute. Default: False.
            recompute_i: bool.
                If true, recompute the ODEs for i^k. If false,
                load the data for i^k if they exist in storage.
                Otherwise, compute. Default: False.
            recompute_k_sym: bool.
                If true, recompute the symbolic equations for K^k. If false,
                load the symbolic equations if they exist in storage.
                Otherwise, compute. Default: False.
            recompute_p_sym: bool.
                If true, recompute the symbolic equations for p^k. If false,
                load the symbolic equations if they exist in storage.
                Otherwise, compute. Default: False.
            recompute_k_sym: bool.
                If true, recompute the symbolic equations for H^k. If false,
                load the symbolic equations if they exist in storage.
                Otherwise, compute. Default: False.
            recompute_h: bool.
                If true, recompute the H functions for H^k. If false,
                load the data equations if they exist in storage.
                Otherwise, compute. Default: False.
            g_forward: list or bool.
                If bool, integrate forwards or backwards
                when computing g^k. If list, integrate g^k forwards or
                backwards based on bool value g_forward[k].
                Default: False.
            z_forward: list or bool.
                Same idea as g_forward for PRCS. Default: False.
            i_forward: list or bool.
                Same idea as g_forward for IRCS. Default: False.
            dense: bool.
                If True, solve_ivp uses dense=True and evaluate solution
                along tLC.
            dir: str.
                Location of data directory. Please choose carefully
                because some outputs may be on the order of gigabytes
                if NA >= 5000. Write 'home+data_dir/' to save to the folder
                'data_dir' in the home directory. Otherwise the script
                will use the current working directory by default uless
                an absolute path is used. The trailing '/' is
                required. Default: None.
            trunc_order: int.
                Highest order to truncate the expansion. For example, 
                trunc_order = 3 means the code will compute up to and 
                including order 3. Default: 3.
            NA: int.
                Number of partitions to discretize phase when computing p.
                Default: 500.
            p_iter: int.
                Number of periods to integrate when computing the time 
                interal in p. Default: 10.
            max_iter: int.
                Number of Newton iterations. Default: 20.
            TN: int.
                Total time steps when computing g, z, i.
            rtol, atol: float.
                Relative and absolute tolerance for ODE solvers.
                Defaults: 1e-7, 1e-7.
            rel_tol: float.
                Threshold for use in Newton scheme. Default: 1e-6.
            method: string.
                Specify the method used in scipy.integrate.solve_ivp.
                Default: LSODA.
            g_bad_dx: list or bool. If bool, use another variable to increase
                the magnitude of the Newton derivative. This can only be
                determined after attempting to run simulations and seeing that
                the Jacobian for the Newton step is ill-conditioned. If list,
                check for ill-conditioning for each order k.
                For example, we use g_small_dx = [False,True,False,...,False]
                for the thalamic model. The CGL model only needs
                g_small_idx = False
            z_Gbad_idx: same idea as g_small_idx for PRCs
            i_bad_idx: same idea as g_small_idx for IRCs
        
        """
        defaults = {
            'trunc_order':3,
            'trunc_deriv':3,

            'TN':20000,
            'dir':None,
            'LC_long_sim_time':500,
            'LC_eps_time':1e-4,
            'LC_tol':1e-13,

            'NA':500,
            'p_iter':10,
            'max_iter':100,

            'rtol':1e-7,
            'atol':1e-7,
            'rel_tol':1e-6,
            'method':'LSODA',
            'g_forward':True,
            'z_forward':True,
            'i_forward':True,

            'g_bad_dx':False,
            'z_bad_dx':False,
            'i_bad_dx':False,

            'dense':False,
            'escale':1,

            'g_jac_eps':1e-3,
            'z_jac_eps':1e-3,
            'i_jac_eps':1e-3,

            'coupling_pars':'',

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

            'processes':2,
            'chunksize':10000
            }

        self.rhs = rhs
        self.coupling = coupling
        self.LC_init = LC_init

        self.rule_par = {}

        # if no kwarg for default, use default. otherwise use input kwarg.
        for (prop, default) in defaults.items():
            value = kwargs.get(prop, default)
            setattr(self, prop, value)

        assert((type(self.g_forward) is bool) or\
               (type(self.g_forward) is list))
        assert((type(self.z_forward) is bool) or\
               (type(self.z_forward) is list))
        assert((type(self.i_forward) is bool) or
               (type(self.i_forward) is list))

        assert((type(self.g_jac_eps) is float) or\
               (type(self.g_jac_eps) is list))
        assert((type(self.z_jac_eps) is float) or\
               (type(self.z_jac_eps) is list))
        assert((type(self.i_jac_eps) is float) or\
               (type(self.i_jac_eps) is list))

        # update self with model parameters and save to dict
        self.pardict_sym = {}
        self.pardict_val = {}
        for (prop, value) in pardict.items():

            # define sympy names, and parameter replacement rule.
            if prop.split('_')[-1] == 'val':
                parname = prop.split('_')[0]

                # save parname_val
                setattr(self,prop,value)

                # sympy name using parname
                symvar = symbols(parname)
                setattr(self,parname,symvar)

                # define replacement rule for parameters
                # i.e. parname (sympy) to parname_val (float/int)
                self.rule_par.update({symvar:value})
                self.pardict_sym.update({parname:symvar})
                self.pardict_val.update({parname:value})


        # variable names
        self.var_names = var_names
        self.dim = len(self.var_names)

        # max iter number
        self.miter = self.trunc_order+1

        # Symbolic variables and functions
        self.eye = np.identity(self.dim)
        self.psi, self.eps, self.kappa = sym.symbols('psi eps kappa')


        # single-oscillator variables and coupling variadef bles.
        # single oscillator vars use the names from var_names
        # A and B are appended to coupling variables
        # to denote oscillator 1 and 2.

        self.vars = []
        self.A_vars = []
        self.B_vars = []

        self.dA_vars = []
        self.dB_vars = []

        #self.A_pair = Matrix([[self.vA,self.hA,self.rA,self.wA,
        #                       self.vB,self.hB,self.rB,self.wB]])
        self.A_pair = sym.zeros(1,2*self.dim)
        self.B_pair = sym.zeros(1,2*self.dim)

        self.dA_pair = sym.zeros(1,2*self.dim)
        self.dB_pair = sym.zeros(1,2*self.dim)

        self.dx_vec = sym.zeros(1,self.dim) 
        self.x_vec = sym.zeros(self.dim,1)

        #Matrix([[self.dv,self.dh,self.dr,self.dw]])
        #Matrix([[self.v],[self.h],[self.r],[self.w]])

        for i,name in enumerate(var_names):
            # save var1, var2, ..., varN
            symname = symbols(name)
            setattr(self, name, symname)
            self.vars.append(symname)
            self.x_vec[i] = symname

            # save dvar1, dvar2, ..., dvarN
            symd = symbols('d'+name)
            setattr(self, 'd'+name, symd)
            self.dx_vec[i] = symd

            # save var1A, var2A, ..., varNA,
            #      var1B, var2B, ..., varNB
            symA = symbols(name+'A')
            symB = symbols(name+'B')
            setattr(self, name+'A', symA)
            setattr(self, name+'B', symB)
            self.A_vars.append(symA)
            self.B_vars.append(symB)

            self.A_pair[:,i] = Matrix([[symA]])
            self.A_pair[:,i+self.dim] = Matrix([[symB]])

            self.B_pair[:,i] = Matrix([[symB]])
            self.B_pair[:,i+self.dim] = Matrix([[symA]])

            symdA = symbols('d'+name+'A')
            symdB = symbols('d'+name+'B')
            setattr(self, 'd'+name+'A', symdA)
            setattr(self, 'd'+name+'B', symdB)

            self.dA_vars.append(symdA)
            self.dB_vars.append(symdB)

            self.dA_pair[:,i] = Matrix([[symdA]])
            self.dA_pair[:,i+self.dim] = Matrix([[symdB]])

            self.dB_pair[:,i] = Matrix([[symdB]])
            self.dB_pair[:,i+self.dim] = Matrix([[symdA]])



        self.t = symbols('t')
        self.tA, self.tB = symbols('tA tB')


        #self.dv, self.dh, self.dr, self.dw = symbols('dv dh dr dw')

        # coupling variables
        self.thA, self.psiA = symbols('thA psiA')
        self.thB, self.psiB = symbols('thB psiB')

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

        #from os.path import expanduser
        #home = expanduser("~")

        # filenames and directories
        if self.dir is None:
            raise ValueError('Please define a data directory using \
                             the keyword argument \'dir\'.\
                             Write dir=\'home+file\' to save to file in the\
                             home directory. Write dir=\'file\' to save to\
                             file in the current working directory.')

        elif self.dir.split('+')[0] == 'home':
            from pathlib import Path
            home = str(Path.home())
            self.dir = home+'/'+self.dir.split('+')[1]

        else:
            self.dir = self.dir

        print('Saving data to '+self.dir)

        if (not os.path.exists(self.dir)):
            os.makedirs(self.dir)

        if self.coupling_pars == '':
            print('NOTE: coupling_pars set to default empty string.'\
                  +'Please specify coupling_pars in kwargs if'\
                  +'varying parameters.')

        lib.generate_fnames(self,coupling_pars=self.coupling_pars)

        # make rhs callable
        #self.rhs_sym = self.thal_rhs(0,self.vars,option='sym')
        self.rhs_sym = rhs(0,self.vars,self.pardict_sym,
                                option='sym')

        #print('jac sym',self.jac_sym[0,0])
        self.load_limit_cycle()

        self.A_array,self.dxA = np.linspace(0,self.T,self.NA,
                                            retstep=True,
                                            endpoint=True)

        self.Aarr_noend,self.dxA_noend = np.linspace(0,self.T,self.NA,
                                                     retstep=True,
                                                     endpoint=False)

        if self.load_all:

            slib.generate_expansions(self)
            slib.load_coupling_expansions(self)
            slib.load_jac_sym(self)

            rule = {**self.rule_LC,**self.rule_par}

            print('jac sym',self.jac_sym)
            print('jac sym subs',self.jac_sym.subs(rule))
            print('rule',rule)
            # callable jacobian matrix evaluated along limit cycle
            self.jacLC = lambdify((self.t),self.jac_sym.subs(rule),
                                  modules='numpy')

            start = time.time();print(self.jacLC(20));end = time.time();print('jac eval time',end-start)

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
        
        
    def monodromy(self,t,z):
        """
        calculate right-hand side of system
        
        
        $\dot \Phi = J\Phi, \Phi(0)=I$,
        
        where $\Phi$ is a matrix solution
        
        jacLC is the jacobian evaluated along the limit cycle
        """
        
        jac = self.jacLC(t)
        #LC_vec = np.array([self.LC['lam_v'](t),
        #                   self.LC['lam_h'](t),
        #                   self.LC['lam_r'](t),
        #                   self.LC['lam_w'](t)])
        
        #jac = self.numerical_jac(rhs,self.LC_vec(t))
        
        #print(jac)
        n = int(np.sqrt(len(z)))
        z = np.reshape(z,(n,n))
        
        #print(n)
        dy = np.dot(jac,z)
        
        return np.reshape(dy,n*n)

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
        
    

    def generate_expansions(self):
        """
        generate expansions from Wilson 2020
        """
        i_sym = sym.symbols('i_sym')  # summation index
        psi = self.psi
        
        #self.g_expand = {}
        for key in self.var_names:
            sg = Sum(psi**i_sym*Indexed('g'+key,i_sym),(i_sym,1,self.miter))
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
        self.rule_d2g = {self.d[k]:
                         self.g['expand_'+k] for k in self.var_names}
        
        #print('self.rule_d2g)',self.rule_d2g)
        #print('rule_d2g',self.rule_d2g)
        
    def load_limit_cycle(self):
        
        self.LC['dat'] = []
        
        for key in self.var_names:
            self.LC['imp_'+key] = []
            self.LC['lam_'+key] = []
            
            print('* Computing LC data...')
        file_does_not_exist = not(os.path.isfile(self.LC['dat_fname']))
        
        #print(os.path.isfile(self.LC['dat_fname']))
        if self.recompute_LC or file_does_not_exist:
            # get limit cycle (LC) period
            sol,t_arr = self.generate_limit_cycle()
            
            # save LC data 
            np.savetxt(self.LC['dat_fname'],sol)
            np.savetxt(self.LC['t_fname'],t_arr)
            
        else:
            #print('loading LC')
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
        imp_lc = sym.zeros(1,self.dim)
        for i,key in enumerate(self.var_names):
            fn = interpb(self.LC['t'],self.LC['dat'][:,i],self.T)
            #fn = interp1d(self.LC['t'],self.LC['dat'][:,i],self.T,kind='cubic')
            self.LC['imp_'+key] = imp_fn(key,fn)
            self.LC['lam_'+key] = fn
            
            imp_lc[i] = self.LC['imp_'+key](self.t)
            #lam_list.append(self.LC['lam_'+key])
            
        self.LC_vec = lambdify(self.t,imp_lc,modules='numpy')
        #self.LC_vec = lam_vec(lam_list)
            
        if True:
            fig, axs = plt.subplots(nrows=self.dim,ncols=1)
            print('LC init',end=', ')
            
            for i, ax in enumerate(axs.flat):
                key = self.var_names[i]
                ax.plot(self.tLC,self.LC['lam_'+key](self.tLC))
                print(self.LC['lam_'+key](0),end=', ')
            axs[0].set_title('LC')
            
            plt.tight_layout()
            plt.savefig('plot_LC.png')
            plt.close()
            #plt.show(block=True)
            
        
        # single rule
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
        
        tol = self.LC_tol
        
        #T_init = 5.7
        eps = np.zeros(self.dim) + 1e-2
        epstime = self.LC_eps_time
        dy = np.zeros(self.dim+1)+10

        # rough init found using XPP
        init = self.LC_init
        T_init = init[-1]
        
        # run for a while to settle close to limit cycle
        sol = solve_ivp(self.rhs,[0,self.LC_long_sim_time],init[:-1],
                        method=self.method,dense_output=True,
                        rtol=1e-13,atol=1e-13,args=(self.pardict_val,))
        
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sol.t,sol.y.T[:,0])
            plt.savefig('plot_limit_cycle_long.png')
        
        tn = len(sol.y.T)
        
        # find max index after ignoring first 20% of simulation time
        # avoids transients.
        maxidx = np.argmax(sol.y.T[int(.2*tn):,0])+int(.2*tn)

        
        init = np.append(sol.y.T[maxidx,:],T_init)
        
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
                                 method=self.method,
                                 rtol=1e-13,atol=1e-13,
                                 #dense_output=True,
                                 #t_eval=t,
                                 args=(self.pardict_val,))
                
                solm = solve_ivp(self.rhs,[0,t[-1]],initm,
                                 method=self.method,
                                 rtol=1e-13,atol=1e-13,
                                 #dense_output=True,
                                 #t_eval=t,
                                 args=(self.pardict_val,))
            
            
                
                yp = solp.y.T
                ym = solm.y.T

                J[:-1,p] = (yp[-1,:]-ym[-1,:])/(2*eps[p])
                
                
            
            J[:-1,:-1] = J[:-1,:-1] - np.eye(self.dim)
            
            
            tp = np.linspace(0,init[-1]+epstime,self.TN)
            tm = np.linspace(0,init[-1]-epstime,self.TN)
            
            # get error in time estimate
            solp = solve_ivp(self.rhs,[0,tp[-1]],initp,
                             method=self.method,
                             rtol=1e-13,atol=1e-13,
                             #dense_output=True,
                             #t_eval=tp,
                             args=(self.pardict_val,))
            
            solm = solve_ivp(self.rhs,[0,tm[-1]],initm,
                             method=self.method,
                             rtol=1e-13,atol=1e-13,
                             #dense_output=True,
                             #t_eval=tm,
                             args=(self.pardict_val,))
            
            yp = solp.y.T
            ym = solm.y.T
            
            J[:-1,-1] = (yp[-1,:]-ym[-1,:])/(2*epstime)
            
            J[-1,:] = np.append(self.rhs(0,init[:-1],self.pardict_val),0)
            #print(J)
            
            sol = solve_ivp(self.rhs,[0,init[-1]],init[:-1],
                             method=self.method,
                             rtol=1e-13,atol=1e-13,
                             #dense_output=True,
                             #t_eval=t,
                             args=(self.pardict_val,))
            
            y_final = sol.y.T[-1,:]
            
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

        print('lc init',sol.y.T[peak_idx,:])
        
        # run finalized limit cycle solution
        sol = solve_ivp(self.rhs,[0,init[-1]],sol.y.T[peak_idx,:],
                        method=self.method,
                        t_eval=np.linspace(0,init[-1],self.TN),
                        rtol=1e-13,atol=1e-13,
                        args=(self.pardict_val,))
        
        #print('warning: lc init set by hand')
        #sol = solve_ivp(self.thal_rhs,[0,init[-1]],init[:-1],
        #                method='LSODA',
        #                t_eval=np.linspace(0,init[-1],self.TN),
        #                rtol=self.rtol,atol=self.atol)
            
        return sol.y.T,sol.t


    def load_monodromy(self):
        """
        if monodromy data exists, load. if DNE or 
        recompute required, compute here.
        """
        
        
        if self.recompute_monodromy\
            or not(os.path.isfile(self.monodromy_fname)):
            
            initm = copy.deepcopy(self.eye)
            r,c = np.shape(initm)
            init = np.reshape(initm,r*c)

            start = time.time();
            sol = solve_ivp(self.monodromy,[0,self.tLC[-1]],init,
                            t_eval=self.tLC,
                            method=self.method,
                            rtol=1e-13,atol=1e-13)

            end = time.time();print('mon eval time',end-start)
            
            self.sol = sol.y.T
            self.M = np.reshape(self.sol[-1,:],(r,c))
            np.savetxt(self.monodromy_fname,self.M)
            
            if False:
                fig, axs = plt.subplots(nrows=r,ncols=c)
                
                for i in range(axs[:,0]):
                    for j in range(axs[0,:]):
                        axs[i,j].plot(self.tLC,sol.y.T[i,j])
                    
                axs[0,1].set_title('monodromy')
                #print('monodromy'+' ini',sol.y.T[0,:])
                #print('g'+str(i)+' fin',data[-1,:])
                plt.tight_layout()
                plt.savefig('monodromy.png')
                plt.close()
            
            

        else:
            self.M = np.loadtxt(self.monodromy_fname)

        print('monodromy',self.M)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.M)
        
        print('eigenvals',self.eigenvalues)
        print(np.log(self.eigenvalues)/self.T)
        
        # get smallest eigenvalue and associated eigenvector
        self.min_lam_idx = np.argsort(self.eigenvalues)[-2]
        #print(self.min_lam_idx)
        #print(self.eigenvalues[self.min_lam_idx])
        self.lam = self.eigenvalues[self.min_lam_idx]  # floquet mult.
        self.kappa_val = np.log(self.lam)/self.T  # floquet exponent
        
        if np.sum(self.eigenvectors[:,self.min_lam_idx]) < 0:
            self.eigenvectors[:,self.min_lam_idx] *= -1

        print('evec',self.eigenvectors)
        
        #print('eigenvalues',self.eigenvalues)
        #print('eiogenvectors',self.eigenvectors)
        
        #print(self.eigenvectors)
        
        # print floquet multipliers
        
        
        einv = np.linalg.inv(self.eigenvectors*self.escale)
        #print('eig inverse',einv)

        
        idx = np.argsort(np.abs(self.eigenvalues-1))[0]
        #min_lam_idx2 = np.argsort(einv)[-2]
        
        
            
        self.g1_init = self.eigenvectors[:,self.min_lam_idx]*self.escale
        self.z0_init = einv[idx,:]
        self.i0_init = einv[self.min_lam_idx,:]
        
        #print('min idx for prc',idx,)
        
        #print('Monodromy',self.M)
        #print('eigenvectors',self.eigenvectors)
        
        print('g1_init',self.g1_init)
        print('z0_init',self.z0_init)
        print('i0_init',self.i0_init)
        
        #print('Floquet Multiplier',self.lam)
        print('* Floquet Exponent kappa =',self.kappa_val)
        
        
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
            print('* Computing g symbolic...')
            
            # create symbolic derivative
            sym_collected = slib.generate_g_sym(self)  
            
            for i in range(self.miter):
                for key in self.var_names:
                    expr = sym_collected[key].coeff(self.psi,i)
        
        
                    self.g['sym_'+key].append(expr)
                    dill.dump(self.g['sym_'+key][i],
                              open(self.g['sym_fnames_'+key][i],'wb'),
                              recurse=True)

                    
        else:
            print('* Loading g symbolic...')
            for key in self.var_names:
                self.g['sym_'+key] = lib.load_dill(self.g['sym_fnames_'+key])
        
    
    def load_g(self):
        """
        load all Floquet eigenfunctions g or recompute
        """
        
        self.g['dat'] = []
        
        for key in self.var_names:
            self.g['imp_'+key] = []
            self.g['lam_'+key] = []
        
        print('* Computing g...')
        for i in range(self.miter):
            print(str(i))
            fname = self.g['dat_fnames'][i]
            
            file_does_not_exist = not(os.path.isfile(fname))
            if self.recompute_g or file_does_not_exist:
                
                het_vec = self.interp_lam(i,self.g,fn_type='g')
                
                data = self.generate_g(i,het_vec)
                np.savetxt(self.g['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            print(len(data),len(self.tLC))
            if True:
                fig, axs = plt.subplots(nrows=self.dim,ncols=1)
                
                for j,ax in enumerate(axs):
                    key = self.var_names[j]
                    ax.plot(self.tLC,data[:,j],label=key)
                    ax.legend()
                    
                axs[0].set_title('g'+str(i))
                print('g'+str(i)+' ini',data[0,:])
                print('g'+str(i)+' fin',data[-1,:])
                plt.tight_layout()
                plt.savefig('plot_g'+str(i)+'.png')
                plt.close()
                
                
            self.g['dat'].append(data)
            
            for j,key in enumerate(self.var_names):
                fn = interpb(self.tLC,data[:,j],self.T)
                #fn = interp1d(self.tLC,data[:,j],self.T,kind='cubic')
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
                
    
    def generate_g(self,k,het_vec):
        """
        generate Floquet eigenfunctions g
        
        uses Newtons method
        """
        
        if type(self.g_forward) is bool:
            backwards = not(self.g_forward)
        elif type(self.g_forward) is list:
            backwards = not(self.g_forward[k])
        else:
            raise ValueError('g_forward must be bool or list, not',
                             type(self.g_forward))
            
        
        if type(self.g_jac_eps) is float:
            eps = self.g_jac_eps
        elif type(self.g_jac_eps) is list:
            eps= self.g_jac_eps[k]
        else:
            raise ValueError('g_jac_eps must be float or list or floats, not',
                             type(self.g_jac_eps))
        
        # load kth expansion of g for k >= 0
        if k == 0:
            # g0 is 0. do this to keep indexing simple.
            return np.zeros((self.TN,len(self.var_names)))
        
        if k == 1:
            # pick correct normalization
            init = copy.deepcopy(self.g1_init)
            eps = 1e-4
        else:
            init = np.zeros(self.dim)
            eps = 1e-4
            init = lib.run_newton2(self,self._dg,init,k,het_vec,
                                  max_iter=self.max_iter,eps=eps,
                                  rel_tol=self.rel_tol,rel_err=10,
                                  alpha=1,backwards=backwards,
                                  dense=self.dense)
        
        # get full solution
        if backwards:
            tLC = -self.tLC
            
        else:
            tLC = self.tLC
        
        print(len(tLC),len(self.tLC))
        sol = solve_ivp(self._dg,[0,tLC[-1]],
                        init,args=(k,het_vec),
                        t_eval=tLC,
                        method=self.method,
                        dense_output=True,
                        rtol=self.rtol,atol=self.atol)
        
        
        print(len(tLC),len(self.tLC),tLC[-1],sol.t[-1])
        if backwards:
            gu = sol.y.T[::-1,:]
            
        else:
            gu = sol.y.T
        print(len(sol.y.T),len(sol.t))
        return gu


    def load_het_sym(self):
        # load het. for z and i if they exist. otherwise generate.
        for key in self.var_names:
            self.z['sym_'+key] = []
            self.i['sym_'+key] = []
        
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
            print('* Computing heterogeneous terms...')
            sym_collected = self.generate_het_sym()
            
            for i in range(self.miter):
                for key in self.var_names:
                    
                    expr = sym_collected[key].coeff(self.psi,i)
                    self.z['sym_'+key].append(expr)
                    self.i['sym_'+key].append(expr)
                    
                    dill.dump(self.z['sym_'+key][i],
                              open(self.z['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    dill.dump(self.i['sym_'+key][i],
                              open(self.i['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                
            # save matrix of a_i
            dill.dump(self.A,open(self.A_fname,'wb'),recurse=True)
            
        else:
            print('* Loading heterogeneous terms...')
            self.A, = lib.load_dill([self.A_fname])
            for key in self.var_names:
                self.z['sym_'+key] = lib.load_dill(self.z['sym_fnames_'+key])
                self.i['sym_'+key] = lib.load_dill(self.i['sym_fnames_'+key])
                
                print('')
                print(key,self.z['sym_'+key])
        
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
            print('z,i het sym deriv order=',i)
            p1 = lib.kProd(i,self.dx_vec)
            p2 = kp(p1,sym.eye(self.dim))

            for j,key in enumerate(self.var_names):
                print('\t var=',key)
                d1 = lib.vec(lib.df(self.rhs_sym[j],self.x_vec,i+1))
                self.a[key] += (1/math.factorial(i))*(p2*d1)
                
                
        self.A = sym.zeros(self.dim,self.dim)
        
        for i,key in enumerate(self.var_names):            
            self.A[:,i] = self.a[key]
        
        het = self.A*self.z['vec']
        
        # expand all terms
        out = {}
        
        rule = {**self.rule_g0,**self.rule_d2g}
        
        rule_trunc = {}
        for k in range(self.miter,self.miter+200):
            rule_trunc.update({self.psi**k:0})
            
        for i,key in enumerate(self.var_names):
            print('z,i het sym subs key=',key)
            tmp = het[i].subs(rule)
            tmp = sym.expand(tmp,basic=False,deep=True,
                             power_base=False,power_exp=False,
                             mul=False,log=False,
                             multinomial=True)
            
            tmp = tmp.subs(rule_trunc)
            tmp = sym.collect(tmp,self.psi).subs(rule_trunc)
            tmp = sym.expand(tmp).subs(rule_trunc)
            tmp = sym.collect(tmp,self.psi).subs(rule_trunc)
            
            out[key] = tmp
            
        return out
    
        
    def load_z(self):
        """
        load all PRCs z or recompute
        """
        
        self.z['dat'] = []
        
        for key in self.var_names:
            self.z['imp_'+key] = []
            self.z['lam_'+key] = []
            
        print('* Computing z...')
        for i in range(self.miter):
            print(str(i))
            
            fname = self.z['dat_fnames'][i]
            file_does_not_exist = not(os.path.isfile(fname))
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
                
                print('z'+str(i)+' ini',data[0,:])
                print('z'+str(i)+' fin',data[-1,:])
                axs[0].set_title('z'+str(i))
                plt.tight_layout()
                plt.savefig('plot_z'+str(i)+'.png')
                plt.close()
                #time.sleep(.1)
                    
            self.z['dat'].append(data)
            
            for j,key in enumerate(self.var_names):
                fn = interpb(self.tLC,data[:,j],self.T)
                #fn = interp1d(self.tLC,data[:,j],self.T,kind='cubic')
                imp = imp_fn('z'+key+'_'+str(i),self.fmod(fn))
                self.z['imp_'+key].append(imp)
                self.z['lam_'+key].append(fn)
                
        
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
        if type(self.z_forward) is bool:
            backwards = not(self.z_forward)
        elif type(self.z_forward) is list:
            backwards = not(self.z_forward[k])
        else:
            raise ValueError('z_forward must be bool or list, not',
                             type(self.z_forward))
            
        
        
        if type(self.z_jac_eps) is float:
            eps = self.z_jac_eps
        elif type(self.z_jac_eps) is list:
            eps= self.z_jac_eps[k]
        else:
            raise ValueError('z_jac_eps must be bool or list, not',
                             type(self.z_jac_eps))
        
        
        
        if k == 0:
            init = copy.deepcopy(self.z0_init)
            #init = [-1.389, -1.077, 9.645, 0]
        else:
            init = np.zeros(self.dim)
            
            init = lib.run_newton2(self,self._dz,init,k,het_vec,
                                  max_iter=self.max_iter,eps=eps,alpha=1,
                                  rel_tol=self.rel_tol,rel_err=10,
                                  backwards=backwards,dense=self.dense)
        
        if backwards:
            tLC = -self.tLC
            
        else:
            tLC = self.tLC
            
        sol = solve_ivp(self._dz,[0,tLC[-1]],
                        init,args=(k,het_vec),
                        method=self.method,dense_output=True,
                        t_eval=tLC,
                        rtol=self.rtol,atol=self.atol)
        
        if backwards:
            zu = sol.y.T[::-1,:]
            
        else:
            zu = sol.y.T
        
        if k == 0:
            # normalize
            dLC = self.rhs(0,self.LC_vec(0)[0],self.pardict_val)
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
        
        print('* Computing i...')
        for i in range(self.miter):
            print(str(i))
            fname = self.i['dat_fnames'][i]
            file_does_not_exist = not(os.path.isfile(fname))
            
            if self.recompute_i or file_does_not_exist:
                
                het_vec = self.interp_lam(i,self.i,fn_type='i')
                data = self.generate_i(i,het_vec)
                np.savetxt(self.i['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)
                
            if True:
                fig, axs = plt.subplots(nrows=self.dim,ncols=1)
                
                for j,ax in enumerate(axs):
                    key = self.var_names[j]
                    ax.plot(self.tLC,data[:,j],label=key)
                    ax.legend()
                    
                print('i'+str(i)+' ini',data[0,:])
                print('i'+str(i)+' fin',data[-1,:])
                axs[0].set_title('i'+str(i))
                plt.tight_layout()
                plt.savefig('plot_i'+str(i)+'.png')
                plt.close()
                
            self.i['dat'].append(data)
            
            for j,key in enumerate(self.var_names):
                fn = interpb(self.tLC,data[:,j],self.T)
                #fn = interp1d(self.tLC,data[:,j],self.T,kind='linear')
                imp = imp_fn('i'+key+'_'+str(i),self.fmod(fn))
                self.i['imp_'+key].append(imp)
                self.i['lam_'+key].append(fn)
                
                #lam_temp = lambdify(self.t,self.i['imp_'+key][i](self.t))
                
        
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
        
    
    
    def generate_i(self,k,het_vec):
        """
        i0 equation is stable in forwards time
        i1, i2, etc equations are stable in backwards time.

        """
        
        if type(self.i_forward) is bool:
            backwards = not(self.i_forward)
        elif type(self.i_forward) is list:
            backwards = not(self.i_forward[k])
        else:
            raise ValueError('i_forward must be bool or list, not',
                             type(self.i_forward))
        
        if type(self.i_bad_dx) is bool:
            exception = self.i_bad_dx
        elif type(self.i_bad_dx) is list:
            exception = self.i_bad_dx[k]
        else:
            raise ValueError('i_bad_dx must be bool or list, not',
                             type(self.i_bad_dx))
            
        if type(self.i_jac_eps) is float:
            eps = self.i_jac_eps
        elif type(self.i_jac_eps) is list:
            eps= self.i_jac_eps[k]
        else:
            raise ValueError('i_jac_eps must be bool or list, not',
                             type(self.i_jac_eps))
        
        if k == 0:
            init = copy.deepcopy(self.i0_init)
        else:
            
            if k == 1:
                alpha = 1
            else:
                alpha = 1
            init = np.zeros(self.dim)
            init = lib.run_newton2(self,self._di,init,k,het_vec,
                                   max_iter=self.max_iter,rel_tol=self.rel_tol,
                                   eps=eps,alpha=alpha,
                                   backwards=backwards,
                                   exception=exception,
                                   dense=self.dense)

        if backwards:
            tLC = -self.tLC
            
        else:
            tLC = self.tLC
        
        sol = solve_ivp(self._di,[0,tLC[-1]],init,
                        args=(k,het_vec),
                        t_eval=tLC,
                        method=self.method,dense_output=True,
                        rtol=self.rtol,atol=self.atol)
        
        if backwards:
            iu = sol.y.T[::-1,:]
            
        else:
            iu = sol.y.T
        
        print('i init',k,iu[0,:])
        print('i final',k,iu[-1,:])
        
        if k == 0:
            # normalize. classic weak coupling theory normalization
            c = np.dot(self.g1_init,iu[0,:])
            iu /= c
    
        if k == 1:  # normalize
        
            # kill off nonzero v
            #if np.sum(self.g['dat'][1][:,-1]) < 1e-20:
            #    iu[:,-1] = 0
            
            # see Wilson 2020 PRE for normalization formula.
            LC0 = []
            g10 = []
            z00 = []
            i00 = []
        
            for varname in self.var_names:
                key = 'lam_'+varname
                LC0.append(self.LC[key](0))
                g10.append(self.g[key][1](0))
                z00.append(self.z[key][0](0))
                i00.append(self.i[key][0](0))
                
            F = self.rhs(0,LC0,self.pardict_val)
            g1 = np.array(g10)
            z0 = np.array(z00)
            i0 = np.array(i00)
            
            J = self.jacLC(0)
            i1 = iu[0,:]
            
            ijg = np.dot(i0,np.dot(J,g1))
            be = (self.kappa_val - ijg - np.dot(i1,F))/(np.dot(z0,F))
            
            init = iu[0,:] + be*z0
            
            sol = solve_ivp(self._di,[0,tLC[-1]],init,
                            args=(k,het_vec),
                            t_eval=tLC,
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
            print('* Computing K symbolic...')
            
            self.cA, self.cB = self.generate_k_sym()
            
            self.cA['vec'] = sym.zeros(self.dim,1)
            self.cB['vec'] = sym.zeros(self.dim,1)
            
            for i,key in enumerate(self.var_names):
                self.cA['vec'][i] = self.cA[key]
                self.cB['vec'][i] = self.cB[key]
            
            # dump
            dill.dump(self.cA['vec'],
                      open(self.cA['sym_fname'],'wb'),recurse=True)
            dill.dump(self.cB['vec'],
                      open(self.cB['sym_fname'],'wb'),recurse=True)
            
            
            rule_trunc = {}
            for k in range(self.miter,self.miter+500):
                rule_trunc.update({self.eps**k:0})
            
            for key in self.var_names:
                tmp = expand(self.cA[key],basic=False,deep=True,
                             power_base=False,power_exp=False,
                             mul=False,log=False,
                             multinomial=True)
                tmp = tmp.subs(rule_trunc)
                
                collectedA = collect(tmp,self.eps)
                collectedA = collect(expand(collectedA),self.eps)
                
                tmp = expand(self.cB[key],basic=False,deep=True,
                             power_base=False,power_exp=False,
                             mul=False,log=False,
                             multinomial=True)
                tmp = tmp.subs(rule_trunc)
                
                collectedB = collect(tmp,self.eps)
                collectedB = collect(expand(collectedB),self.eps)
                
                self.cA[key+'_col'] = collectedA
                self.cB[key+'_col'] = collectedB
                
                
                for i in range(self.miter):
                    
                    # save each order to list and dill.
                    eps_i_termA = self.cA[key+'_col'].coeff(self.eps,i)
                    eps_i_termB = self.cB[key+'_col'].coeff(self.eps,i)
                        
                    
                    self.kA['sym_'+key].append(eps_i_termA)
                    self.kB['sym_'+key].append(eps_i_termB)
                    
                    dill.dump(self.kA['sym_'+key][i],
                              open(self.kA['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    
                    dill.dump(self.kB['sym_'+key][i],
                              open(self.kB['sym_fnames_'+key][i],'wb'),
                              recurse=True)
                    
                
        else:
            print('* Loading K symbolic...')
            for key in self.var_names:
                self.kA['sym_'+key] = lib.load_dill(self.kA['sym_fnames_'+key])
                self.kB['sym_'+key] = lib.load_dill(self.kB['sym_fnames_'+key])
            
            self.cA['vec'] = lib.load_dill([self.cA['sym_fname']])[0]
            self.cB['vec'] = lib.load_dill([self.cB['sym_fname']])[0]

        for i in range(len(self.kA['sym_x'])):
            print(i,': ',self.kA['sym_x'][i])
            
    def generate_k_sym(self):
        # generate terms involving the coupling term (see K in paper).
        
        # find K_i^{j,k}
        #coupA = self.thal_coupling(self.A_pair,option='sym')
        #coupB = self.thal_coupling(self.B_pair,option='sym')

        coupA = self.coupling(self.A_pair,self.pardict_sym,option='sym')
        coupB = self.coupling(self.B_pair,self.pardict_sym,option='sym')

        # get expansion for coupling term

        # 0 and 1st derivative
        for i,key in enumerate(self.var_names):
            self.cA[key] = coupA[i]
            self.cA[key] += lib.df(coupA[i],self.A_pair,1).dot(self.dA_pair)
            
            self.cB[key] = coupB[i]
            self.cB[key] += lib.df(coupB[i],self.B_pair,1).dot(self.dB_pair)
            
        # 2nd + derivative
        
        for i in range(2,self.trunc_deriv+1):
            # all x1,x2 are evaluated on limit cycle x=cos(t), y=sin(t)
            kA = lib.kProd(i,self.dA_pair)
            kB = lib.kProd(i,self.dB_pair)
            
            for j,key in enumerate(self.var_names):
                dA = lib.vec(lib.df(coupA[j],self.A_pair,i))
                dB = lib.vec(lib.df(coupB[j],self.B_pair,i))
                
                self.cA[key] += (1/math.factorial(i))*kA.dot(dA)
                self.cB[key] += (1/math.factorial(i))*kB.dot(dB)
        
        rule = {}
        for i,key in enumerate(self.var_names):
            rule.update({self.dA_vars[i]:self.g[key+'_epsA']})
            rule.update({self.dB_vars[i]:self.g[key+'_epsB']})
            
        for key in self.var_names:
            self.cA[key] = self.cA[key].subs(rule)
            self.cB[key] = self.cB[key].subs(rule)
            
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
            print('* Computing p symbolic...')
            ircA = self.eps*self.i['vecA'].dot(self.cA['vec'])
            
            rule_trunc = {}
            for k in range(self.miter,self.miter+500):
                rule_trunc.update({self.eps**k:0})
        
            tmp = expand(ircA,basic=False,deep=True,
                         power_base=False,power_exp=False,
                         mul=False,log=False,
                         multinomial=True)
            tmp = tmp.subs(rule_trunc)
            
            ircA = collect(tmp,self.eps)
            ircA = collect(expand(ircA),self.eps)
            
            for i in range(self.miter):
                # save each order to list and dill.
                eps_i_termA = ircA.coeff(self.eps,i)
                self.pA['sym'].append(eps_i_termA)

                dill.dump(self.pA['sym'][i],
                          open(self.pA['sym_fnames'][i],'wb'),recurse=True)

        else:
            print('* Loading p symbolic...')
            self.pA['sym'] = lib.load_dill(self.pA['sym_fnames'])
        

    def load_p(self):
        """
        generate/load the ODEs for psi.
        """
        
        
        # load all p or recompute or compute new.
        self.pA['dat'] = []
        self.pA['imp'] = []
        self.pA['lam'] = []
        
        
        self.pool = _ProcessPool(processes=self.processes)
        
        
        for i,fname in enumerate(self.pA['dat_fnames']):
            
            if self.recompute_p or not(os.path.isfile(fname)):
                print('* Computing p'+str(i)+'...')
                start = time.time()
                
                old = 0
                if old:
                    pA_data = self.generate_p_old(i)
                else:
                    pA_data = self.generate_p(i)
                end = time.time()
                print('time elapsed for p'+str(i)+' = '+str(end-start))
                
                np.savetxt(self.pA['dat_fnames'][i],pA_data)
                
            else:
                print('* Loading p'+str(i)+'...')
                pA_data = np.loadtxt(fname)
                
            
            if True:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                
                ax.imshow(pA_data,cmap='viridis',aspect='auto')
                                
                ax.set_ylabel('B')
                ax.set_xlabel('A')
                ax.set_title('pA data'+str(i)\
                             +' NA='+str(self.NA)\
                             +' piter='+str(self.p_iter))
                #plt.show(block=True)
                plt.savefig('figs_temp/p_'+str(i)+'.png')
                plt.close()
            
            pA_interp = interp2d(self.Aarr_noend,
                                 self.Aarr_noend,
                                 pA_data,bounds_error=False,
                                 fill_value=None,kind='cubic')
            
            pA_interp = interp2db(pA_interp,self.T)
            
            pA_imp = imp_fn('pA_'+str(i),self.fLam2(pA_interp))
            
            self.pA['dat'].append(pA_data)
            
            if i == 0:
                imp = imp_fn('pA_0', lambda x: 0)
                self.pA['imp'].append(imp)
                self.pA['lam'].append(0)
            
            else:
                self.pA['imp'].append(pA_imp)
                self.pA['lam'].append(pA_interp)
                
        self.pool.close()
        self.pool.join()
        self.pool.terminate()
        
        
        ta = self.thA
        tb = self.thB
        
        
        rule_pA = {sym.Indexed('pA',i):self.pA['imp'][i](ta,tb)
                       for i in range(self.miter)}
        rule_pB = {sym.Indexed('pB',i):self.pA['imp'][i](tb,ta)
                       for i in range(self.miter)}
        
        self.rule_p_AB = {**rule_pA,**rule_pB}
        
        
    def generate_p(self,k):
        #import scipy as sp
        import numpy as np
        
        kappa = self.kappa_val
        
        ta = self.thA
        tb = self.thB
        
        NA = self.NA
        dxA_noend = self.dxA_noend
        #dxA = self.dxA
        T = self.T
        
        p = self.pool
        
        if k == 0:
            #pA0 is 0 (no forcing function)
            return np.zeros((self.NA,self.NA))
        
        # put these implemented functions into the expansion
        ruleA = {sym.Indexed('pA',i):
                 self.pA['imp'][i](ta,tb) for i in range(k)}
        ruleB = {sym.Indexed('pB',i):
                 self.pA['imp'][i](tb,ta) for i in range(k)}
        
        
        rule = {**ruleA, **ruleB,**self.rule_g_AB,
                **self.rule_i_AB,**self.rule_LC_AB,
                **self.rule_par}
        
        ph_impA = self.pA['sym'][k].subs(rule)
        repl, redu = sym.cse(ph_impA)
        
        funs = []
        syms = [ta,tb]
        for i, v in enumerate(repl):
            funs.append(lambdify(syms,v[1],modules='numpy'))
            syms.append(v[0])
        
        glam = lambdify(syms,redu[0],modules='numpy')
        
        lam_hetA_data = np.zeros((NA,NA))
        
        A_array2 = self.Aarr_noend
        B_array2 = self.Aarr_noend

        print('compile 0')
        for i in range(NA):
            if i % 1000 == 0:
                print(i)
            ta2 = A_array2[i]*np.ones_like(B_array2)
            tb2 = B_array2
            
            xs = [ta2,tb2]
            
            for f in funs:
                xs.append(f(*xs))
        
            #lam_hetA_data[:,i] = lam_hetA(ta2,tb2)
            lam_hetA_data[:,i] = glam(*xs)

        fig,axs = plt.subplots()
        im = axs.imshow(lam_hetA_data)
        plt.colorbar(im,ax=axs)
        plt.savefig('figs_temp/lam_heta_'+str(k)+'.png')
        plt.close()
        
        # pg 184 brandeis notebook
        # u \in [-\infty 0] and (0,theta_1)
        
        exp = np.exp        
        p_iter = self.p_iter
        u1 = np.arange(0,-T*p_iter,-dxA_noend)
        #u1 = np.arange(0,-T*p_iter,-dxA)
        u1_idxs = np.arange(0,-len(u1),-1,dtype=int)
        exponential1 = exp(-kappa*u1)
        
        # integral 1
        i1 = np.zeros(NA)
        
        def return_integral1(i):
            pass
        
        print('compile 1')
        for i in range(NA):
            p_idxs = np.remainder(i+u1_idxs,NA)
            periodic = lam_hetA_data[p_idxs,np.remainder(u1_idxs,NA)]
            i1[i] = np.sum(exponential1*periodic)*dxA_noend
        
        # integral 2
        i2 = np.zeros((NA,NA))
        
        u2 = A_array2
        u2_idxs = np.arange(len(u2))
        exponential2 = exp(-kappa*u2)
        
        print('compile 2')
        """  
        print('compile 2')
        for i in range(NA):
            for j in range(NA):
                #th1 = u2[i]
                #th2 = A_array2[j]
                
                #xs = [u2[:i+1],th2-th1+u2[:i+1]]
                
                #for f in funs:
                #    xs.append(f(*xs))
                
                #periodic = lam_hetA(u2[:i+1],th2-th1+u2[:i+1])#glam(*xs)
                #periodic = glam(*xs)
                
                p_idxs = np.remainder(j-i + u2_idxs[:i+1],NA)
                periodic = lam_hetA_data[p_idxs,u2_idxs[:i+1]]
                #print(i,j)
                i2[i,j] = np.sum(exponential2[:i+1]*periodic)*dxA_noend
                #print('i2,ij',i,j,
                #      np.sum(exponential[:i+1]*periodic)*dxA_noend,
                #      exponential[:i+1],
                #      p_idxs)
                #print(i,j,th1,th2,i2[i,j])
        """
        
        i2 = np.reshape(i2,(NA**2,))
        
        a_i = np.arange(NA,dtype=int)
        b_i = np.arange(NA,dtype=int)
        
        A_mg_idxs, B_mg_idxs = np.meshgrid(a_i,b_i,indexing='ij')
        
        a_mg_idxs = np.reshape(A_mg_idxs,(NA*(NA),))
        b_mg_idxs = np.reshape(B_mg_idxs,(NA*(NA),))
        
        idx = np.arange(len(a_mg_idxs))
        
        def return_integral(i):
            th1_idx = a_mg_idxs[i]
            th2_idx = b_mg_idxs[i]
            
            #th1 = u2[th1_idx]
            #th2 = A_array2[th2_idx]
            
            #xs = [u2[:th1_idx+1],th2-th1+u2[:th1_idx+1]]
            
            #for f in funs:
            #    xs.append(f(*xs))
            
            p_idxs = np.remainder(th2_idx-th1_idx+ u2_idxs[:th1_idx+1],NA)
            periodic = lam_hetA_data[p_idxs,u2_idxs[:th1_idx+1]]
            #print(i,j)
            #print('i2,ij',i,j,
            #      np.sum(exponential[:i+1]*periodic)*dxA_noend,
            #      exponential[:i+1],
            #      p_idxs)
            #print(i,j,th1,th2,i2[i,j])
            
            #periodic = lam_hetA(u2[:th1_idx+1],th2-th1+u2[:th1_idx+1])
            #return np.sum(exp(-kappa*u2[:th1_idx+1])*periodic)*dxA_noend, i
            #return np.sum(exp(-kappa*u2[:th1_idx+1])*periodic)*dxA, i
            #p_idxs = np.remainder(th2_idx-th1_idx + u_idxs[:th1_idx],NA)
            #periodic = lam_hetA_data[p_idxs,u_idxs[:th1_idx]]
            #print(i,j)
            
            return np.sum(exponential2[:th1_idx+1]*periodic)*dxA_noend, i
            #return np.sum(exponential2[:th1_idx+1]*periodic)*dxA, i
            
         
        
        
        for x in tqdm.tqdm(p.imap(return_integral,idx,
                                  chunksize=self.chunksize),
                           total=len(idx)):
            integral, idx = x
            i2[idx] = integral
        
        i2 = np.reshape(i2,(NA,NA))
        
            
        
        pA_data = np.zeros((NA,NA))
        """
        expo = np.exp(-kappa*u)
        #print('compile 3')
        for i in range(NA):
            for j in range(NA):
                
                p_idxs = np.remainder(j-i + u_idxs,NA)
                periodic = lam_hetA_data[p_idxs,np.remainder(u_idxs,NA)]
                
                phi_idx = j-i #np.abs(diff_idx)
                
                int1 = i1[phi_idx]
                
                int2 = i2[i,j]
                
                pA_data[j,i] = exp(kappa*th1)*(int1+int2)
        
        
        """
        #print('compile 3')
        for i in range(NA):
            for j in range(NA):
                th1 = A_array2[i]
                #th2 = A_array2[j]
                #diff_idx = 
                phi_idx = j-i #np.abs(diff_idx)
                
                int1 = i1[phi_idx]
                
                int2 = i2[i,j]
                
                pA_data[j,i] = exp(kappa*th1)*(int1+int2)
        
        #print(pA_data)
        
        """
        pA_data = np.reshape(pA_data,(NA**2,))
        
        a_i = np.arange(NA,dtype=int)
        A_mg_idxs, B_mg_idxs = np.meshgrid(a_i,a_i)
        
        a_mg_idxs = np.reshape(A_mg_idxs,(NA**2,))
        b_mg_idxs = np.reshape(B_mg_idxs,(NA**2,))
        
        idx = np.arange(len(a_mg_idxs))
        
        
        def return_integral(i):
            #return time integral at position a,b
            th1_idx = a_mg_idxs[i]
            th2_idx = b_mg_idxs[i]
            th1 = A_array2[th1_idx]
            #th2 = A_array2[th2_idx]
            
            phi_idx = th2_idx-th1_idx
            
            int1 = i1[phi_idx]
            
            if th1_idx > 0:
                int2 = i2[th1_idx-1,th2_idx]
            else:
                int2 = 0
    
            return exp(kappa*th1)*(int1+int2), i
    
        
        p = self.pool
        
        for x in tqdm.tqdm(p.imap(return_integral,idx,chunksize=1000),
                           total=len(idx)):
            integral, idx = x
            pA_data[idx] = integral
            
        pA_data = np.reshape(pA_data,(NA,NA))
        """
        
        return pA_data
    
        
    def generate_p_old(self,k):
        
        #import scipy as sp
        import numpy as np
        #import time
        
        ta = self.thA
        tb = self.thB
        
        NA = self.NA
        dxA_noend = self.dxA_noend
        T = self.T
        
        if k == 0:
            #pA0 is 0 (no forcing function)
            return np.zeros((self.NA,self.NA))
        
        # put these implemented functions into the expansion
        ruleA = {sym.Indexed('pA',i):
                 self.pA['imp'][i](ta,tb) for i in range(k)}
        ruleB = {sym.Indexed('pB',i):
                 self.pA['imp'][i](tb,ta) for i in range(k)}
        
        
        rule = {**ruleA, **ruleB,**self.rule_g_AB,
                **self.rule_i_AB,**self.rule_LC_AB,
                **self.rule_par}
        
        ph_impA = self.pA['sym'][k].subs(rule)
        
        # this lambidfy calls symbolic functions. slow.
        # convert lamdify to data and call linear interpolation on that.
        # then function call is same speed independent of order.
        #lam_hetA = lambdify([ta,tb],ph_impA,modules='numpy')
        #lam_hetA_old = lam_hetA
        
        # https://stackoverflow.com/questions/30738840/...
        # best-practice-for-using-common-subexpression-elimination...
        # -with-lambdify-in-sympy
        repl, redu = sym.cse(ph_impA)
        
        funs = []
        syms = [ta,tb]
        for i, v in enumerate(repl):
            funs.append(lambdify(syms,v[1],modules='numpy'))
            syms.append(v[0])
        
        glam = lambdify(syms,redu[0],modules='numpy')
        
        #lam_hetA = lambdify([ta,tb],ph_impA)
        
        lam_hetA_data = np.zeros((NA,NA))
        
        A_array2 = self.Aarr_noend
        B_array2 = self.Aarr_noend
        
        
        for i in range(NA):
            ta2 = A_array2[i]*np.ones_like(B_array2)
            tb2 = B_array2
            
            xs = [ta2,tb2]
            
            for f in funs:
                xs.append(f(*xs))
        
            #lam_hetA_data[:,i] = lam_hetA(ta2,tb2)
            lam_hetA_data[:,i] = glam(*xs)
        
        
        A_mg, B_mg = np.meshgrid(A_array2,B_array2)
        
        
        # parallelize
        kappa = self.kappa_val
       
        r,c = np.shape(A_mg)
        a = np.reshape(A_mg,(r*c,))

        # get indices where lam_hetA_data is above threshold
        #lam_dat_reshaped = np.reshape(lam_hetA_data,(r*c,))

        a_i = np.arange(NA,dtype=int)
        A_mg_idxs, B_mg_idxs = np.meshgrid(a_i,a_i)
        
        a_mg_idxs = np.reshape(A_mg_idxs,(r*c,))
        b_mg_idxs = np.reshape(B_mg_idxs,(r*c,))
        
        
        pA_data = np.zeros((r,c))
        pA_data = np.reshape(pA_data,(r*c,))
        
        idx = np.arange(len(a))
        exp = np.exp
        
        p_iter = self.p_iter
        
        s = np.arange(0,T*p_iter,dxA_noend)
        s_idxs = np.arange(len(s),dtype=int)
        exponential = exp(s*kappa)
        
        
        def return_integral(i):
            """
            return time integral at position a,b
            """
            a_idxs = np.remainder(a_mg_idxs[i]-s_idxs,NA)
            b_idxs = np.remainder(b_mg_idxs[i]-s_idxs,NA)
            
            #keepA = np.isin(a_idxs,sigA)
            #keepB = np.isin(b_idxs,sigB)
            
            #coincide = keepA*keepB
            
            #print(keepB)
            #print(i,a_idxs,b_idxs)
            
            #a_idxs = a_mg_idxs[i]-s_idxs
            #b_idxs = b_mg_idxs[i]-s_idxs
            slice = lam_hetA_data[b_idxs,a_idxs]
            #keep_idxs = np.abs(slice)/magnitude>1e-4
            periodic = slice#[keep_idxs]
            #expo = exponential[keep_idxs]
            integrand = exponential*periodic
            #integrand = integrand[integrand/magnitude>1e-24]
            #fn = lam_hetA_data[b_idxs[coincide],a_idxs[coincide]]
            #expo = exponential[coincide]
            #tot = np.sum(exponential*periodic)*dxA_noend
            #tot = np.add.reduce(expo*fn)*dxA_noend

            #return tot, i
            #return np.add.reduce(exponential*periodic)*dxA_noend, i
            return np.add.reduce(integrand)*dxA_noend, i
        
        
        #start = time.time()
        #for i in range(100):
        #    return_integral2(10)
        #end = time.time()
        #print('time elapsed for integral'+str(i)+' = '+str(end-start))
        #pA_data = self.generate_p_old(i)
        
        #for i in range(len(a)):
        #    integral, idx = return_integral(i)
        #    pA_data[idx] = integral
        
        p = self.pool
        
        #for res in p.imap_unordered(return_integral,idx,chunksize=100000):
        #    integral, idx = res
        #    pA_data[idx] = integral
        
        
        for x in tqdm.tqdm(p.imap_unordered(return_integral,idx,
                                            chunksize=10000),
                           total=len(a)):
            integral, idx = x
            pA_data[idx] = integral
            #pA_data += x
        
        pA_data = np.reshape(pA_data,(r,c))
        
        #print(pA_data)
        
        return pA_data
    
    def load_h_sym(self):
        """
        also compute h lam
        """
        
        # symbolic h terms
        self.hodd['sym'] = []
        
        # simplify expansion for speed
        rule_trunc = {}
        for k in range(self.miter,self.miter+200):
            rule_trunc.update({self.eps**k:0})
        
        
        #print(lib.files_exist(self.hodd['sym_fnames']))
        if self.recompute_h_sym or \
            not(lib.files_exist(self.hodd['sym_fnames'])):
            print('* Computing H symbolic...')
            
            z_rule = {}
            for key in self.var_names:
                for i in range(self.miter):
                    z_rule.update({Indexed('z'+key,i):
                                   Indexed('z'+key+'A',i)})

            z_rule.update({self.psi:self.pA['expand']})
            
            z = self.z['vec'].subs(z_rule)
            dotproduct = self.cA['vec'].dot(z)
            
            tmp = sym.expand(dotproduct,basic=False,deep=True,
                             power_base=False,power_exp=False,
                             mul=False,log=False,
                             multinomial=True)
            
            tmp = tmp.subs(rule_trunc)
            tmp = expand(tmp).subs(rule_trunc)
            self.h_collected = collect(tmp,self.eps)
            
            #collected = collect(expand(dotproduct),self.eps)
            #self.h_collected = collect(expand(collected),self.eps)
            
            for i in range(self.miter):
                collected = self.h_collected.coeff(self.eps,i)
                
                # try collecting similar terms
                #collected = sym.cse(collected,optimization='basic')
                
                #print(collected)
                self.hodd['sym'].append(collected)
                dill.dump(self.hodd['sym'][i],
                          open(self.hodd['sym_fnames'][i],'wb'),recurse=True)
                
        else:
            print('* Loading H symbolic...')
            self.hodd['sym'] = lib.load_dill(self.hodd['sym_fnames'])
            
        # lambdify symbolic call
        self.hodd['lam'] = []
        
        # change lam into data set and interp2d call for speed.
        
        rule = {**self.rule_p_AB,**self.rule_g_AB,
                **self.rule_z_AB,**self.rule_LC_AB,
                **self.rule_par}
        
        ta = self.thA
        tb = self.thB
        
        for i in range(self.miter):
            #pass
            h_lam = lambdify([ta,tb],self.hodd['sym'][i].subs(rule),
                             modules='numpy')
            self.hodd['lam'].append(h_lam)
        
            
    def load_h(self):
        
        self.hodd['dat'] = []
        self.hlam = {}
        
        for i in range(self.miter):
            fname = self.hodd['dat_fnames'][i]
            file_does_not_exist = not(os.path.isfile(fname))
            
            if self.recompute_h or file_does_not_exist:
                
                print('* Computing H'+str(i)+'...')
                data = self.generate_h_odd(i)
                np.savetxt(self.hodd['dat_fnames'][i],data)

            else:
                print('* Loading H'+str(i)+'...')
                data = np.loadtxt(self.hodd['dat_fnames'][i])
                
            
                
            self.hodd['dat'].append(data)

        #self.pool.close()
        #self.pool.join()
        #self.pool.terminate()


    def generate_h_odd(self,k):
        """
        interaction functions
        
        note to self: see nb page 130 for notes on indexing in sums.
        need to sum over to index N-1 out of size N to avoid
        double counting boundaries in mod operator.
        """
        
        #T = self.T
        
        h = np.zeros(self.NA)
        
        t = self.A_array
        
        #hodd_lam_k = self.hodd['lam'][k]
        
        rule = {**self.rule_p_AB,**self.rule_g_AB,
                **self.rule_z_AB,**self.rule_LC_AB,
                **self.rule_par}
        
        # https://stackoverflow.com/questions/30738840/...
        # best-practice-for-using-common-subexpression-elimination...
        # -with-lambdify-in-sympy
        #print('hodd sym',self.hodd['sym'][k].subs(rule))
        syms = [self.thA,self.thB]

        repl, redu = sym.cse(self.hodd['sym'][k].subs(rule))
        
        funs = []
        
        for ii, v in enumerate(repl):
            funs.append(lambdify(syms,v[1],modules='numpy'))
            syms.append(v[0])
        
        glam = lambdify(syms,redu[0],modules='numpy')
        
        #lam_h_data[:,j] = glam(*xs)

        for j in range(self.NA):
            eta = self.A_array[j]

            xs = [t,t+eta]
            for f in funs:
                xs.append(f(*xs))
                
            h[j] = np.sum(glam(*xs))*self.dxA/self.T


        #print('h, i',k,self.hodd['sym'][k].subs(rule))

        #for j in range(self.NA):
        #    eta = self.A_array[j]
        #    xs = [t,t+eta]
        #    
        #    glam = lambdify(syms,self.hodd['sym'][k].subs(rule))
        #
        #    h[j] = np.sum(glam(*xs))*self.dxA/self.T
        
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(h)
            title = ('h non-odd '+str(k)
                     +', NA='+str(self.NA)
                     +', piter='+str(self.p_iter))
            ax.set_title(title)
            plt.savefig('plot_h_non_odd'+str(k)+'.png')
            plt.close()
        
        #print(h)
        #hodd = h
        hodd = (h[::-1]-h)
        
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(hodd)
            title = ('h odd '+str(k)
                     +', NA='+str(self.NA)
                     +', piter='+str(self.p_iter))
            ax.set_title(title)
            plt.savefig('plot_hodd'+str(k)+'.png')
            plt.close()
        
        return hodd
            
    def _dg(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        #print(t)
        
        jac = self.jacLC(t)*(order > 0)
        hom = np.dot(jac-order*self.kappa_val*self.eye,z.T)
        out = hom + het_vec(t)
    
    
        return out
    
    def _dz(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = np.dot(self.jacLC(t).T+order*self.kappa_val*self.eye,z.T)
        out = -hom - het_vec(t)
        
        return out
    
    def _di(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jacLC: jacobian on LC
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = np.dot(self.jacLC(t).T+(order-1)*self.kappa_val*self.eye,z)
        out = -hom - het_vec(t)
        
        return out
    
    def interp_lam(self,k,fn_dict,fn_type='z'):
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
        
        # z and i use the same heterogeneous terms
        # so they are indistinguishable in this function.
        if fn_type == 'z' or fn_type == 'i':
            fn_type = 'z'
        
        rule = {}
        for key in self.var_names:
            tmp = {sym.Indexed(fn_type+key,i):fn_dict['imp_'+key][i](self.t)
                   for i in range(k)}
            
            rule.update(tmp)
        
        rule = {**rule,**self.rule_LC,**self.rule_par}
        if fn_type == 'z':
            rule.update({**self.rule_g})
        
        het_imp = sym.zeros(1,self.dim)
        for i,key in enumerate(self.var_names):
            sym_fn = fn_dict['sym_'+key][k].subs(rule)
            
            lam = lambdify(self.t,sym_fn,modules='numpy')
            
            # evaluate
            if fn_type == 'g' and (k == 0 or k == 1):
                y = np.zeros(self.TN)
            elif fn_type == 'z' and k == 0:
                y = np.zeros(self.TN)
            elif fn_type == 'i' and k == 0:
                y = np.zeros(self.TN)
            elif sym_fn == 0:
                y= np.zeros(self.TN)
            else:
                y = lam(self.tLC)
            
            #print(y,k,self.tLC,sym_fn)
            if False:
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(lam(2*self.tLC))
                ax.set_title('het_lam'+key)
                plt.show(block=True)
            
            # save as implemented fn
            interp = interpb(self.LC['t'],y,self.T)
            imp = imp_fn(key,self.fmod(interp))
            het_imp[i] = imp(self.t)
            
            
        het_vec = lambdify(self.t,het_imp,modules='numpy')
        
        if False and k > 0:
            fig, axs = plt.subplots(nrows=self.dim,ncols=1)
            for i,key in enumerate(self.var_names):
                
                axs[i].plot(self.tLC*2,het_vec[key](self.tLC*2))
            
            axs[0].set_title('lam dict')
            plt.tight_layout()
            plt.show(block=True)
            
        return het_vec
    
    def fmod(self,fn):
        """
        fn has mod built-in
        
        input function-like. usually interp1d object
        
        needed to keep lambda input variable unique to fn.
        
        otherwise lambda will use the same input variable for 
        all lambda functions.
        """
        return lambda x=self.t: fn(x)
    
    def fLam2(self,fn):
        """
        interp2db object
        """
        return lambda xA=self.thA,xB=self.thB: fn(xA,xB)
        
    def bispeu(self,fn,x,y):
        """
        silly workaround
        https://stackoverflow.com/questions/47087109/\
        evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
        """
        return si.dfitpack.bispeu(fn.tck[0], fn.tck[1],
                                  fn.tck[2], fn.tck[3],
                                  fn.tck[4], x, y)[0]


