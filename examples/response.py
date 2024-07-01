"""
calculate limit cycle,
monodromy,
and response functions.


todo: need to check accuracy of response functions
"""


import lib.lib_sym2 as slib
from lib import lib2 as lib
from lib import fnames
from lib.fast_interp import interp1d
from lib.util import get_period

import os
import logging
import numpy as np

import time
import dill
import copy

import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sympy as sym
from sympy import Matrix, symbols, Sum, Indexed, collect, expand
from sympy import sympify
from sympy.utilities.lambdify import lambdify, implemented_function

from sympy.physics.quantum import TensorProduct as kp
import math

imp_fn = implemented_function

class Response(object):
    
    def __init__(self,var_names,
                 
                 rhs,
                 init,
                 TN,
                 idx,

                 coupling=None,

                 pardict=None,
                 forcing_fn=None,
                 
                 dir_root='./data',
                 model_name=None,
                 pars_for_fname='',
                 
                 recompute_list=[],
                 method='LSODA',
                 log_level='CRITICAL',
                 log_file='log_response.log',
                 
                 trunc_order=2,
                 g_forward=True,
                 z_forward=False,
                 i_forward=False,
                 
                 g_jac_eps=1e-3,
                 z_jac_eps=1e-3,
                 i_jac_eps=1e-3,
                 i_bad_dx=False,

                 rtol=1e-10,
                 atol=1e-10,
                 max_iter=30,
                 rel_tol=1e-10,

                 save_fig=False,
                 factor=1,

                 mode='1:1'):
            
        var_names = copy.deepcopy(var_names)
        pardict = copy.deepcopy(pardict)

        self.factor = 1 # scale response functions
        
        self.mode = mode
        self.save_fig = save_fig
        # if forcing, period is assumed known.
        self.forcing = False
        if not(forcing_fn is None):
            self.forcing_fn = forcing_fn
            self.forcing = True

            self.syms = [symbols('f'+str(idx))]
            self.dsyms = [symbols('df'+str(idx))]

        self.idx = idx

        self.pars_for_fname = pars_for_fname
        
        self.max_iter = max_iter
        self.rtol = rtol
        self.atol = atol
        self.rel_tol = rel_tol
        
        self.g_jac_eps = g_jac_eps
        self.z_jac_eps = z_jac_eps
        self.i_jac_eps = i_jac_eps
        
        self.g_forward = g_forward
        self.z_forward = z_forward
        self.i_forward = i_forward

        self.i_bad_dx = i_bad_dx
        
        self.trunc_order = trunc_order
        self.miter = self.trunc_order+1
        
        self.log_level = log_level
        self.log_file = log_file

        if self.log_level == 'DEBUG':
            self.log_level = logging.DEBUG
        elif self.log_level == 'INFO':
            self.log_level = logging.INFO
        elif self.log_level == 'WARNING':
            self.log_level = logging.WARNING
        elif self.log_level == 'ERROR':
            self.log_level = logging.ERROR
        elif self.log_level == 'CRITICAL':
            self.log_level = logging.CRITICAL

        FORMAT = '%(asctime)s %(message)s'
        logging.basicConfig(filename=self.log_file,level=self.log_level,
                            format=FORMAT)

        for i in range(len(var_names)):
            var_names[i] += str(self.idx)
        self.var_names = var_names
        self.dim = len(self.var_names)
        
        keys = list(pardict.keys())

        for key in keys:
            pardict[key+str(idx)] = pardict.pop(key)
        
        self.pardict = pardict
        self.rhs = rhs
        self.coupling = coupling
        
        self.init = init
        self.TN = TN
        self.dir_root = dir_root
        self.recompute_list = recompute_list
        
        self.method = method

        self.rule_par = {}
        self.pardict_sym = {}
        for (prop, value) in pardict.items():

            parname = prop
            symvar = symbols(parname)

            # define replacement rule for parameters
            # i.e. parname (sympy) to parname_val (float/int)
            self.rule_par.update({prop:value})
            self.pardict_sym.update({prop:symvar})
            
        assert(not(model_name is None))

        self.model_name = model_name

        ########### create directories
        if not(os.path.isdir(self.dir_root)):
            os.mkdir(self.dir_root)

        self.dir1 = self.dir_root+'/'+self.model_name+'/'

        if not(os.path.isdir(self.dir1)):
            os.mkdir(self.dir1)

        ########## define dicts
        self.lc = {};self.g = {}
        self.z = {};self.i = {}
        self.G = {};self.K = {}
        self.p = {};self.h = {}
        self._H0 = {} # for mean values

        self.t = symbols('t',real=True)

        self.eye = np.identity(self.dim)
        self.psi, self.eps, self.kappa = sym.symbols('psi eps kappa')

        self.syms = []; self.dsyms = []
        for j,name in enumerate(var_names):
            self.syms.append(symbols(name))
            self.dsyms.append(symbols('d'+name))

        #self.x_vec = sym.Matrix(self.syms)
        #self.dx_vec = sym.Matrix(self.dsyms)
        #### load fnames
        fnames.load_fnames_response(self,model_pars=self.pars_for_fname)

        ########## calculate stuff
        self.load_lc()


        slib.generate_expansions(self)
        slib.load_coupling_expansions(self)

        # make rhs callable
        self.rhs_sym = rhs(0,self.syms,self.pardict_sym,
                           option='sym',idx=self.idx)
        slib.load_jac_sym(self) # callable jac

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
            
    def load_lc(self):

        self.lc['dat'] = []
                
        for key in self.var_names:
            self.lc['imp_'+key] = []
            self.lc['lam_'+key] = []

        file_dne = not(os.path.isfile(self.lc_fname))

        if 'lc' in self.recompute_list or file_dne:
            print('* Computing limit cycle data...')
            #logging.info('* Computing LC data...')
            y,t = self.generate_lc()

            nr,nc = np.shape(y)
            z = np.zeros([nr,nc+1])
            z[:,0] = t
            z[:,1:] = y

            np.savetxt(self.lc_fname,z)

        else:
            print('* Loading limit cycle data...')
            #logging.info('* Loading LC data...')
            z = np.loadtxt(self.lc_fname)

        # normalize period
        if self.mode == 'nm':
            self.T = 2*np.pi
            
            self.pardict[self.om_fix_key] = z[-1,0]/self.T
            self.rule_par[self.om_fix_key] = z[-1,0]/self.T

            self.T_old = z[-1,0]
            self.lc['t_old'] = z[:,0]
            
            #self.T = z[-1,0]

        else:
            self.T = z[-1,0]

        self.tlc,self.dtlc = np.linspace(0,self.T,self.TN,retstep=True)
        self.lc['t'] = self.tlc
        
        self.lc['dat'] = z[:,1:]
        self.init = z[0,1:]

        print('* Limit cycle period = '+str(self.T))
        logging.debug('* Limit cycle period = '+str(self.T))
                    
        # Make LC data callable from inside sympy
        imp_lc = sym.zeros(self.dim)
        for i,key in enumerate(self.var_names):

            fn = interp1d(self.tlc[0],self.tlc[-1],
                          self.dtlc,self.lc['dat'][:-1,i],p=True,k=9)
            
            imp = imp_fn('lc'+key+'_'+str(i),self.fmod(fn))
            
            self.lc['imp_'+key] = imp_fn(key,fn)
            self.lc['lam_'+key] = fn
            
            imp_lc[i] = self.lc['imp_'+key](self.t)
            #lam_list.append(self.lc['lam_'+key])
            
        self.lc_vec = lambdify(self.t,imp_lc,modules='numpy')
        #self.lc_vec = lam_vec(lam_list)

        if self.save_fig:
            self.save_temp_figure(z[:,1:],0,'LC')

        self.rule_lc_local = {}
        for j,key in enumerate(self.var_names):
            self.rule_lc_local[self.syms[j]] = self.lc['imp_'+key](self.t)

    def generate_lc(self,max_time=5000,method='LSODA',tol_root=1e-13):
        """
        generate limit cycle data for system
        system: dict. 
        """

        eps = np.zeros(self.dim) + 1e-4
        epstime = 1e-4
        dy = np.zeros(self.dim+1) + 10

        init = self.init
        T_init = self.init[-1]

        pardict = self.pardict

        sol = solve_ivp(self.rhs,[0,max_time],
                        self.init[:-1],
                        args=(pardict,'value',self.idx),
                        method=method,
                        dense_output=True,
                        rtol=1e-12,atol=1e-12)

        if self.save_fig:
            if not(os.path.exists('figs_temp')):
                os.makedirs('figs_temp')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sol.t,sol.y.T[:,0])
            ax.set_xlim(sol.t[-1]-(sol.t[-1]/100),sol.t[-1])
            plt.savefig('figs_temp/plot_limit_cycle_long.png')

                
        T_init,res1 = get_period(sol)
        init = np.append(sol.sol(res1),T_init)

        counter = 0
        while np.linalg.norm(dy) > tol_root and\
              counter < self.max_iter:
        
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
                                 args=(pardict,'value',self.idx))

                solm = solve_ivp(self.rhs,[0,t[-1]],initm,
                                 method=self.method,
                                 rtol=1e-13,atol=1e-13,
                                 args=(pardict,'value',self.idx))

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
                             args=(pardict,'value',self.idx),
                             t_eval=tp)

            solm = solve_ivp(self.rhs,[0,tm[-1]],initm,
                             method=self.method,
                             rtol=1e-13,atol=1e-13,
                             args=(pardict,'value',self.idx),
                             t_eval=tm)

            yp = solp.y.T
            ym = solm.y.T

            J[:-1,-1] = (yp[-1,:]-ym[-1,:])/(2*epstime)

            J[-1,:] = np.append(self.rhs(0,init[:-1],pardict,
                                         'value',self.idx),0)

            sol = solve_ivp(self.rhs,[0,init[-1]],init[:-1],
                            method=self.method,
                            rtol=1e-13,atol=1e-13,
                            args=(pardict,'value',self.idx),
                            t_eval=t)

            y_final = sol.y.T[-1,:]
            b = np.append(init[:-1]-y_final,0)

            dy = np.linalg.solve(J,b)
            init += dy

            if False:
                fig,axs = plt.subplots()
                axs.plot(sol.y[0],label='unp')
                axs.plot(solm.y[0],label='pm')
                axs.plot(solp.y[0],label='pp')
                axs.legend()
                
                plt.savefig('figs_temp/iters/lc_'+str(counter)+'.png')


            to_disp = (counter,np.linalg.norm(dy))
            

            str1 = 'iter={}, LC rel. err ={:.2e}     '
            end = '            \r'
            printed = str1.format(counter,np.linalg.norm(dy))
            print(printed,end=end)
            counter += 1

        # find index of peak voltage and initialize.
        peak_idx = np.argmax(sol.y.T[:,0])
        init = sol.y.T[peak_idx,:]

        # run finalized limit cycle solution
        sol = solve_ivp(self.rhs,[0,sol.t[-1]],sol.y.T[peak_idx,:],
                        method=self.method,
                        t_eval=np.linspace(0,sol.t[-1],self.TN),
                        rtol=1e-13,atol=1e-13,
                        args=(pardict,'value',self.idx))

        return sol.y.T,sol.t
    

    def load_monodromy(self):
        """
        if monodromy data exists, load. if DNE or 
        recompute required, compute here.
        """

        file_dne = not(os.path.isfile(self.m_fname))
        if 'm' in self.recompute_list or file_dne:
            print('* Computing monodromy...')
            #logging.info('* Computing monodromy...')

            initm = copy.deepcopy(self.eye)
            r,c = np.shape(initm)
            init = np.reshape(initm,r*c)

            start = time.time();
            sol = solve_ivp(self.monodromy,[0,self.tlc[-1]],init,
                            t_eval=self.tlc,
                            method=self.method,
                            rtol=1e-13,atol=1e-13)
            
            end = time.time();#print('mon eval time',end-start)
            
            self.sol = sol.y.T
            self.M = np.reshape(self.sol[-1,:],(r,c))
            np.savetxt(self.m_fname,self.M)
            
        else:
            print('* Loading monodromy...')
            #logging.info('* Loading monodromy...')
            self.M = np.loadtxt(self.m_fname)
        
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.M)

        logging.debug(str(self.eigenvalues)+str(self.eigenvectors))
        
        # get smallest eigenvalue and associated eigenvector
        self.min_lam_idx = np.argsort(self.eigenvalues)[-2]

        logging.debug('min_lam_idx='+str(self.min_lam_idx))
        logging.debug('eigenstuff'+str(self.eigenvalues[self.min_lam_idx]))
        
        self.lam = self.eigenvalues[self.min_lam_idx]  # floquet mult.
        self.kappa_val = np.log(self.lam)/self.T  # floquet exponent

        # make sign of eigenvectors consistent
        if np.sum(self.eigenvectors[:,self.min_lam_idx]) < 0:
            self.eigenvectors[:,self.min_lam_idx] *= -1
        
        #einv = np.linalg.inv(self.eigenvectors/2)
        einv = np.linalg.inv(self.eigenvectors)
        idx = np.argsort(np.abs(self.eigenvalues-1))[0]
            
        self.g1_init = self.eigenvectors[:,self.min_lam_idx]
        self.z0_init = einv[idx,:]
        self.i0_init = einv[self.min_lam_idx,:]

        logging.debug('eigenvectors'+str(self.eigenvectors))
        
        logging.debug('g1_init'+str(self.g1_init))
        logging.debug('z0_init'+str(self.z0_init))
        logging.debug('i0_init'+str(self.i0_init))

        print('* Floquet Exponent ='+str(self.kappa_val))
        #logging.info('* Floquet Exponent ='+str(self.kappa_val))


    def monodromy(self,t,z):
        """
        calculate right-hand side of system
        
        $\dot \Phi = J\Phi, \Phi(0)=I$,        
        where $\Phi$ is a matrix solution
        
        jaclc is the jacobian evaluated along the limit cycle
        """
        
        jac = self.jaclc(t)
        
        n = int(np.sqrt(len(z)))
        z = np.reshape(z,(n,n))        
        dy = np.dot(jac,z)
        
        return np.reshape(dy,n*n)
    
    def load_g_sym(self):
        # load het. functions h if they exist. otherwise generate.
        #self.rule_g0 = {sym.Indexed('gx',0):s(0),sym.Indexed('gy',0):s(0)}
        
        # create dict of gv0=0,gh0=0,etc for substitution later.
        self.rule_g0 = {sym.Indexed('g'+name,0):
                        sympify(0) for name in self.var_names}

        for key in self.var_names:
            self.g['sym_'+key] = []
        
        # check that files exist
        val = 0
        for key in self.var_names:
            val += not(lib.files_exist(self.g['sym_fnames_'+key]))
        
        if val != 0:
            files_dne = True
        else:
            files_dne = False        
        
        if 'g_sym' in self.recompute_list or files_dne:
            print('* Computing g symbolic...')
            #logging.info('* Computing g symbolic...')
            
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
            #logging.info('* Loading g symbolic...')
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
        #logging.info('* Computing g...')
        for i in range(self.miter):

            fname = self.g['dat_fnames'][i]
            
            file_dne = not(os.path.isfile(fname))
            if 'g'+str(i) in self.recompute_list or file_dne or\
               'g' in self.recompute_list:
                
                het_vec = self.interp_lam(i,self.g,fn_type='g')
                
                data = self.generate_g(i,het_vec)
                np.savetxt(self.g['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)

            data *= self.factor
    
            self.g['dat'].append(data)
            if self.save_fig:
                self.save_temp_figure(data,i,'g')
            
            for j,key in enumerate(self.var_names):
                
                fn = interp1d(self.tlc[0],self.tlc[-1],
                              self.dtlc,data[:-1,j],p=True,k=9)
                imp = imp_fn('g'+key+'_'+str(i),fn)
                
                self.g['imp_'+key].append(imp)
                self.g['lam_'+key].append(fn)                
        
        # replacement rules.        
        self.rule_g_local = {}  # g function

        t = self.t
        for key in self.var_names:
            for k in range(self.miter):
                
                fn_loc = sym.Indexed('g'+key,k)
                d_loc = {fn_loc:self.g['imp_'+key][k](t)}
                self.rule_g_local.update(d_loc) # local

    
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
            eps = 1e-7
        else:
            init = np.zeros(self.dim)
            eps = 1e-7
            init = lib.run_newton2(self,self._dg,init,k,het_vec,
                                  max_iter=self.max_iter,eps=eps,
                                  rel_tol=self.rel_tol,rel_err=10,
                                  alpha=1,backwards=backwards)
        
        # get full solution
        if backwards:
            tlc = -self.tlc
            
        else:
            tlc = self.tlc

        sol = solve_ivp(self._dg,[0,tlc[-1]],
                        init,args=(k,het_vec),
                        t_eval=tlc,
                        method=self.method,
                        rtol=self.rtol,atol=self.atol,
                        dense_output=True)
        
        
        if backwards:
            gu = sol.y.T[::-1,:]
            
        else:
            gu = sol.y.T
        return gu

    def load_z(self):
        """
        load all PRCs z or recompute
        """
        
        self.z['dat'] = []
        
        for key in self.var_names:
            self.z['imp_'+key] = []
            self.z['lam_'+key] = []
            self.z['avg_'+key] = []

        print('* Computing z...')
        #logging.info('* Computing z...')
        for i in range(self.miter):
            
            fname = self.z['dat_fnames'][i]
            file_dne = not(os.path.isfile(fname))
            if 'z'+str(i) in self.recompute_list or file_dne or\
               'z' in self.recompute_list:
                
                het_vec = self.interp_lam(i,self.z,fn_type='z')
                data = self.generate_z(i,het_vec)
                np.savetxt(self.z['dat_fnames'][i],data)
                
            else:
                data = np.loadtxt(fname)

            data *= self.factor

            self.z['dat'].append(data)
            if self.save_fig:
                self.save_temp_figure(data,i,'z')
            
            for j,key in enumerate(self.var_names):

                fn = interp1d(self.tlc[0],self.tlc[-1],
                              self.dtlc,data[:-1,j],p=True,k=9)
                imp = imp_fn('z'+key+'_'+str(i),fn)
                
                self.z['imp_'+key].append(imp)
                self.z['lam_'+key].append(fn)
                self.z['avg_'+key].append(np.mean(data[:,j]))
        
        self.rule_z_local = {}

        # messy but keeps global and local indices clear
        t = self.t
        for key in self.var_names:
            for k in range(self.miter):
                fn = sym.Indexed('z'+key,k)
                d = {fn:self.z['imp_'+key][k](t)}
                self.rule_z_local.update(d) # global

        
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
                                  backwards=backwards)
        
        if backwards:
            tlc = -self.tlc
            
        else:
            tlc = self.tlc
            
        sol = solve_ivp(self._dz,[0,tlc[-1]],
                        init,args=(k,het_vec),
                        method=self.method,
                        t_eval=tlc,
                        rtol=self.rtol,atol=self.atol,
                        dense_output=True)
        
        if backwards:
            zu = sol.y.T[::-1,:]
            
        else:
            zu = sol.y.T
        
        if k == 0:
            # normalize
            dlc = self.rhs(0,self.lc_vec(0)[0],self.pardict,
                           'value',self.idx)
            zu = zu/(np.dot(dlc,zu[0,:]))*2*np.pi/self.T
            
        
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
        #logging.info('* Computing i...')
        for i in range(self.miter):

            fname = self.i['dat_fnames'][i]
            file_dne = not(os.path.isfile(fname))
            
            if 'i'+str(i) in self.recompute_list or file_dne or\
               'i' in self.recompute_list:

                het_vec = self.interp_lam(i,self.i,fn_type='i')
                data = self.generate_i(i,het_vec)
                np.savetxt(fname,data)
                
            else:
                data = np.loadtxt(fname)

            data *= self.factor

            self.i['dat'].append(data)

            if self.save_fig:
                self.save_temp_figure(data,i,'i')
            
            for j,key in enumerate(self.var_names):

                fn = interp1d(self.tlc[0],self.tlc[-1],
                              self.dtlc,data[:-1,j],p=True,k=9)
                imp = imp_fn('i'+key+'_'+str(i),fn)
                
                self.i['imp_'+key].append(imp)
                self.i['lam_'+key].append(fn)
        
        # coupling
        # messy but keeps global and local indices clear

        self.rule_i_local = {}
        t = self.t
        
        for key in self.var_names:
            for k in range(self.miter):                
                fn = sym.Indexed('i'+key,k)
                d = {fn:self.i['imp_'+key][k](t)}
                self.rule_i_local.update(d) # global
        
    
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
            
            init = np.zeros(self.dim)
            init = lib.run_newton2(self,self._di,init,k,het_vec,
                                   max_iter=self.max_iter,rel_tol=self.rel_tol,
                                   eps=eps,alpha=1,backwards=backwards,
                                   exception=exception)

        if backwards:
            tlc = -self.tlc
            
        else:
            tlc = self.tlc
        
        sol = solve_ivp(self._di,[0,tlc[-1]],init,
                        args=(k,het_vec),
                        t_eval=tlc,method=self.method,
                        rtol=self.rtol,atol=self.atol,
                        dense_output=True)
    
        if backwards:
            iu = sol.y.T[::-1,:]
            
        else:
            iu = sol.y.T
                
        if k == 0:
            # normalize. classic weak coupling theory normalization
            c = np.dot(self.g1_init,iu[0,:])
            iu /= c
            
            logging.debug('norm const i0'+str(c))
    
        if k == 1:  # normalize
        
            # kill off nonzero v
            #if np.sum(self.g['dat'][1][:,-1]) < 1e-20:
            #    iu[:,-1] = 0
            
            # see Wilson 2020 PRE for normalization formula.
            lc0 = []
            g10 = []
            z00 = []
            i00 = []
        
            for varname in self.var_names:
                key = 'lam_'+varname
                lc0.append(self.lc[key](0))
                g10.append(self.g[key][1](0))
                z00.append(self.z[key][0](0))
                i00.append(self.i[key][0](0))
                
            F = self.rhs(0,lc0,self.pardict,'value',self.idx)
            g1 = np.array(g10)
            z0 = np.array(z00)
            i0 = np.array(i00)
            
            J = self.jaclc(0)
            i1 = iu[0,:]
            
            ijg = np.dot(i0,np.dot(J,g1))
            be = (self.kappa_val - ijg - np.dot(i1,F))/(np.dot(z0,F))
            
            init = iu[0,:] + be*z0

            sol = solve_ivp(self._di,[0,tlc[-1]],init,
                            args=(k,het_vec),
                            t_eval=tlc,
                            method=self.method,
                            rtol=self.rtol,atol=self.atol,
                            dense_output=True)


            #iu = sol.y.T[::-1,:]
            if backwards:
                iu = sol.y.T[::-1,:]
                
            else:
                iu = sol.y.T

            logging.debug('norm const i1='+str(be))
            #print('norm const i1='+str(be),'iu0',iu[0,:],'z0',z0)
            
        return iu

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
            files_dne = True
        else:
            files_dne = False
            
        
        if 'het' in self.recompute_list or files_dne:
            print('* Computing heterogeneous terms...')
            #logging.info('* Computing heterogeneous terms...')
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
            #logging.info('* Loading heterogeneous terms...')
            self.A, = lib.load_dill([self.A_fname])
            for key in self.var_names:
                self.z['sym_'+key] = lib.load_dill(self.z['sym_fnames_'+key])
                self.i['sym_'+key] = lib.load_dill(self.i['sym_fnames_'+key])

                
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
     
        for i in range(1,self.miter):
            logging.debug('z,i het sym deriv order='+str(i))
            p1 = lib.kProd(i,self.dx_vec)
            p2 = kp(p1,sym.eye(self.dim))

            for j,key in enumerate(self.var_names):
                logging.debug('\t var='+str(key))
                d1 = lib.vec(lib.df(self.rhs_sym[j],self.x_vec,i+1))
                d1 = sym.powsimp(d1)
                self.a[key] += (1/math.factorial(i))*(p2*d1)
                
        self.A = sym.zeros(self.dim,self.dim)
        
        for i,key in enumerate(self.var_names):            
            self.A[:,i] = self.a[key]
        
        het = self.A*self.z['vec_psi']

        # expand all terms
        out = {}        
        rule = {**self.rule_g0,**self.rule_d2g}
        
        rule_trunc = {}
        for k in range(self.miter,self.miter+200):
            rule_trunc.update({self.psi**k:0})
            
        for i,key in enumerate(self.var_names):
            #logging.info('z,i het sym subs key='+str(key))
            
            tmp = het[i].subs(rule)
            tmp = sym.expand(tmp,basic=True,deep=True,
                             power_base=False,power_exp=False,
                             mul=True,log=False,
                             multinomial=True)
            
            tmp = tmp.subs(rule_trunc)
            tmp = sym.collect(tmp,self.psi).subs(rule_trunc)
            tmp = sym.expand(tmp).subs(rule_trunc)
            tmp = sym.collect(tmp,self.psi).subs(rule_trunc)
            tmp = sym.powsimp(tmp)
            out[key] = tmp
            
        return out

    
    def _dg(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jaclc: jacobian on lc
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """

        jac = self.jaclc(t)*(order > 0)
        hom = np.dot(jac-order*self.kappa_val*self.eye,z.T)
        out = hom + het_vec(t).T
    
        return out

    def _dz(self,t,z,order,het_vec):
        """
        g functon rhs with ith het. term
        
        z: position
        t: time
        jaclc: jacobian on lc
        het: heterogeneous terms
        
        order determines the Taylor expansion term
        """
        
        hom = np.dot(self.jaclc(t).T+order*self.kappa_val*self.eye,z.T)
        out = -hom - het_vec(t).T
        
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

        hom = np.dot(self.jaclc(t).T+(order-1)*self.kappa_val*self.eye,z)
        out = -hom - het_vec(t).T
        
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
        if fn_type == 'z' or fn_type == 'i':
            fn_type = 'z'
        
        rule = {}

        for key in self.var_names:
            tmp = {sym.Indexed(fn_type+key,i):fn_dict['imp_'+key][i](self.t)
                   for i in range(k)}
            
            rule.update(tmp)

        
        rule = {**rule,**self.rule_lc_local,**self.rule_par}
        if fn_type == 'z':
            rule.update({**self.rule_g_local})            
            
        het_imp = sym.zeros(self.dim,1)
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
                y = lam(self.tlc)
            
            fn = interp1d(self.tlc[0],self.tlc[-1],self.dtlc,y[:-1],p=True,k=9)
            imp = imp_fn(key,fn)
            
            # save as implemented fn
            het_imp[i] = imp(self.t)
            
        het_vec = lambdify(self.t,het_imp,modules='numpy')
            
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
    
    def save_temp_figure(self,data,k,fn='plot',path_loc='figs_temp/'):
        """
        data should be (TN,dim)
        """

        if (not os.path.exists(path_loc)):
            os.makedirs(path_loc)
        
        fig, axs = plt.subplots(nrows=self.dim,ncols=1)
        
        for j,ax in enumerate(axs):
            key = self.var_names[j]
            ax.plot(self.tlc,data[:,j],label=key)
            ax.legend()

        print(fn+str(k)+' ini'+str(data[0,:]))
        print(fn+str(k)+' fin'+str(data[-1,:]))
              
        #logging.info(fn+str(k)+' ini'+str(data[0,:]))
        #logging.info(fn+str(k)+' fin'+str(data[-1,:]))
        axs[0].set_title(fn+str(k))
        plt.tight_layout()
        plt.savefig(path_loc+fn+str(k)+'_'+self.model_name+'.png')
        plt.close()
        

    def __call__(self,th,option='add'):
        if option == 'add':
            #om = self.pardict['om'+str(self.idx)]
            #a = self.pardict['amp'+str(self.idx)]
            return self.coupling(self.forcing_fn(th),self.pardict)
        
