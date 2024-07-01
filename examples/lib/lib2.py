# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:49:28 2020

@author: youngmin

library functions
"""

import time
import os
import dill
import sys

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

#from scipy.interpolate import interp1d
from sympy.physics.quantum import TensorProduct as kp
#from sympy.utilities.lambdify import lambdify, implemented_function

from scipy.integrate import solve_ivp


def vec(a):
    """
    vec array operator. stack columns.
    
    reshape command stacks rows. so transpose then reshape to stack columns
    
    https://stackoverflow.com/questions/55444777/...
    numpy-array-stack-multiple-columns-into-one-using-reshape
    """
    #print(type(a))
    #a = np.asarray(a)
    
    if np.asarray(a).ndim == 0:
        return a
    else:
        return a.T.reshape(len(a[:,0])*len(a[0,:]),1)
    
    
def grad(fn,xvec):
    """
    fn is a scalar valued function.
    
    xvec is the general input to fn. size of xvec is the dimension of domain
    xevc contains the domain sympy variables [x1,x2,...,xn]
    
    return row vector
    """
    n = len(xvec)
    gradf = sym.zeros(1,n)
    
    for i in range(n):
        gradf[0,i] = sym.diff(fn,xvec[i])
    
    return gradf


def df(fn,xvec,k):
    """
    distinct from grad. we alternate applying vec and grad to fn k times
    f is map from RN to R (see Eq. 13 Wilson 2020)
    
    fn is a function of xvec.
    
    step k=1
    -apply vec to transform gives 1 x 1
    -derivative gives 1 x N
    -end if k=1
    
    step k=2
    -apply vec to previous step gives N x 1
    -deriv gives NxN
    -end if k=2
    
    step k=3
    -apply vec to previous step gives 2*N x 1
    -deriv gives 2*N x N
    -end if k=3
    
    etc.
    
    output size N^(k-1) x N
    
    
    """
    df = fn
    n = len(xvec)
    
    if k == 0:
        return df
    
    if k == 1:
        df = grad(df,xvec)
        return df
    
    # f^(1)
    df = grad(df,xvec)
    #print()
    #print(np.shape(df))
    #print('df1',df)
    
    # proceed with higher derivs
    #print('k',k)
    for i in range(2,k+1):
        #print('i,k',i,k)
        df = vec(df)
        
        # preallocate N^(k-1) x N
        df_temp = sym.zeros(n**(i-1),n)
        
        #print(np.shape(df_temp))
        
        # now loop over rows of df_temp and save gradient
        for j in range(len(df_temp[:,0])):
            df_temp[j,:] = grad(df[j,:],xvec)
        
        df = df_temp
        #print(np.shape(df))
        #print('############ df, i,k',np.shape(df),i,k,df)  
        #print('i,df',i,df)
    
    #print(np.shape(df))
    #print('############ df, k',np.shape(df),k,df)    
    return df
    

def monodromy(t,z,jacLC):
    """
    calculate right-hand side of system
    \dot \Phi = J\Phi, \Phi(0)=I
    \Phi is a matrix solution
    
    jacLC is the jacobian evaluated along the limit cycle
    """
    
    n = int(np.sqrt(len(z)))
    z = np.reshape(z,(n,n))
    
    #print(n)
    dy = np.dot(jacLC(t),z)
    
    return np.reshape(dy,n*n)

def kProd(k,dx):
    """
    Kronecker product applied k times to vector dx (1,n)
    k=1 returns dx
    k=2 returns (1,n^2)
    generally returns (1,n^(k))
    """
    out = dx
    
    for i in range(k-1):
        #print('out',out)
        out = kp(out,dx)
        
    return out


def files_exist(*fnames,dictionary=False):

    
    fname_list = []
    for i in range(len(fnames)):
        fname_list += fnames[i]

    flag = 0
    for i in range(len(fname_list)):
        # check if each fname exists
        flag += not(os.path.isfile(fname_list[i]))
        
    if flag != 0:
        return False
    else:
        return True

def load_dill(fnames):
    #print(fnames)
    templist = []
    for i in range(len(fnames)):
        templist.append(dill.load(open(fnames[i],'rb')))
        
    return templist

def run_newton2(obj,fn,init,k,het_lams,max_iter=10,
                rel_tol=1e-12,rel_err=10,backwards=True,eps=1e-2,
                exception=False,alpha=1,min_iter=5,
                dense=True):
    if backwards:
        tlc = -obj.tlc
    else:
        tlc = obj.tlc
    # run newton's method
    counter = 0
    dx = 100.

    if not(os.path.isdir('figs_temp')):
        os.mkdir('figs_temp')
    
    #smallest_init = np.zeros(len(init))+10
    dx_smallest = np.zeros(len(init))+10.
    init_smallest = init
    
    try:
        while counter < max_iter:
        
            if (np.linalg.norm(dx) < rel_tol) and (counter >= min_iter):
                break
            
            dx_prev = dx
            dx,t,sol = get_newton_jac2(obj,fn,tlc,init,k,het_lams,
                                       return_sol=True,eps=eps,
                                       exception=exception,dense=dense)
            
            if np.linalg.norm(dx_prev) < np.linalg.norm(dx):
                alpha /= 1.1
                
            if np.linalg.norm(dx) < np.linalg.norm(dx_smallest):
                dx_smallest = dx
                init_smallest = init

            init += dx*alpha
            counter += 1
            
            if True:
                fig, axs = plt.subplots(nrows=obj.dim,ncols=1)
                    
                for i,ax in enumerate(axs):
                    key = obj.var_names[i]
                    ax.plot(t,sol[:,i],label=key)
                    ax.legend()
                 
                axs[0].set_title('{} iter {}'.format(str(fn),counter))
                plt.tight_layout()
                #plt.show(block=True)
                #time.sleep(.1)
                args = (obj.model_name,counter)
                fname = 'figs_temp/{}_iter={}.png'.format(*args)
                plt.savefig(fname)
                plt.close()
                
            if True:

                print('dx={:.2e}, al={:.2e}'.format(np.linalg.norm(dx),alpha),
                      end='                   \r')
            
            if counter == max_iter:
                print('')
                print('WARNING: max iter reached in newton call')
                
    except KeyboardInterrupt:
        pass

    print('')
    
    if np.linalg.norm(dx_smallest) < np.linalg.norm(dx):
        return init_smallest
    else:
        return init


def get_newton_jac2(obj,fn,tlc,init,k=None,het_lams=None,return_sol=False,
                    eps=1e-2,exception=False,dense=False):
    """
    Newton derivative. for use in newton's method

    Parameters
    ----------
    init : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if k is not None:
        args = (k,)
        if het_lams is not None:
            args += (het_lams,)
    else:
        args = ()
        
    n = len(init)
    
    J = np.zeros((n,n))
    if dense:
        sol = solve_ivp(fn,[0,tlc[-1]],init,args=args,
                        method=obj.method,
                        rtol=obj.rtol,atol=obj.atol,
                        dense_output=True,t_eval=tlc)
    else:
        sol = solve_ivp(fn,[0,tlc[-1]],init,args=args,
                        method=obj.method,
                        rtol=obj.rtol,atol=obj.atol)
            
    sol_unpert = sol.y.T
    t_unpert = sol.t
    
    n = len(sol_unpert[0,:])
    
    for p in range(len(init)):
            
        pert = np.zeros(len(init))
        pert[p] = eps

        pert_init = init + pert
        
        sol = solve_ivp(fn,[0,tlc[-1]],pert_init,args=args,
                        method=obj.method,rtol=obj.rtol,atol=obj.atol)
                        #dense_output=True, t_eval=tlc)
        sol_pert_p = sol.y.T
        
        J[:,p] = (sol_pert_p[-1,:] - sol_unpert[-1,:])/eps
    
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(obj.tlc,sol_unpert[:,p],color='blue')
            ax.plot(obj.tlc,sol_pert_p[:,p])
            ax.set_title('pert'+str(p))
            
            #ax.set_xlim(0,obj.tlc[-1]/10)
            
            plt.show(block=True)
            time.sleep(2)
            plt.close()
    
    
    mydiff = (sol_unpert[-1,:] - sol_unpert[0,:])
    Jdiff = J - np.identity(n)
        
    if exception:
        v = np.zeros((1,n))
        v[0] = 1
        Jdiff = np.append(Jdiff,v,axis=0)
        mydiff = np.append(mydiff,0)
        dx,residuals,rank,s = np.linalg.lstsq(Jdiff, mydiff,
                                              rcond=None)
    
    else:
        dx = np.linalg.solve(Jdiff, mydiff)

    
    if return_sol:
        return -dx, t_unpert,sol_unpert
    else:
        return dx


def run_newton3(obj,fn,init,k,het_lams,max_iter=10,
                rel_tol=1e-12,rel_err=10,backwards=True,eps=1e-1,
                exception=False,alpha=1,min_iter=5,
                dense=False,jac=None):
    if backwards:
        tlc = -obj.tlc
    else:
        tlc = obj.tlc
    # run newton's method
    counter = 0
    dx = 100.
    dx_smallest = np.zeros(len(init))+10.
    
    try:
        while counter < max_iter:
        
            if (np.linalg.norm(dx) < rel_tol) and (counter >= min_iter):
                break
            
            dx_prev = dx
            dx = get_newton_jac3(obj,fn,tlc,init,k,het_lams,eps=eps,
                                 exception=exception,dense=dense,
                                 jac=jac)
            
            init += dx
            counter += 1
            
            if True:
                print('dx={:.2e}, al={:.2e}'.format(np.linalg.norm(dx),alpha),
                end='\r')
            
            if counter == max_iter:
                print('WARNING: max iter reached in newton call')
                
    except KeyboardInterrupt:
        pass
    
    return init


def get_newton_jac3(obj,fn,tlc,init,k=None,het_lams=None,return_sol=False,
                    eps=1e-1,exception=False,dense=False,jac=None):
    """
    Newton derivative. for use in newton's method

    Parameters
    ----------
    init : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if k is not None:
        args = (k,)
        if het_lams is not None:
            args += (het_lams,)
    else:
        args = ()

    n = len(init)
    
    J = np.zeros((n,n))
    
    sol = solve_ivp(fn,[0,tlc[-1]],init,args=args,
                    method=obj.method,
                    rtol=obj.rtol,atol=obj.atol,
                    dense_output=dense,t_eval=tlc)
    sol_unpert = sol.y.T
    t_unpert = sol.t
    
    n = len(sol_unpert[0,:])
    
    for p in range(len(init)):
        
        pert = np.zeros(len(init))
        pert[p] = eps
        pert_init = init + pert

        sol = solve_ivp(fn,[0,tlc[-1]],pert_init,args=args,
                        method=obj.method,rtol=obj.rtol,atol=obj.atol)
        sol_pert_p = sol.y.T
        
        J[:,p] = (sol_pert_p[-1,:] - sol_unpert[-1,:])/eps
    
    mydiff = sol_unpert[-1,:] - sol_unpert[0,:]
    Jdiff = -(J - np.identity(n))
    
    if exception:
        v = np.zeros((1,n))
        v[0] = 1
        Jdiff = np.append(Jdiff,v,axis=0)
        mydiff = np.append(mydiff,0)
        dx,residuals,rank,s = np.linalg.lstsq(Jdiff, mydiff)
    
    else:
        dx = np.linalg.solve(Jdiff, mydiff)

    return dx


def fname_suffix(exclude=[],ftype='.png',**kwargs):
    """
    generate filename suffix based on parameters
    """
    fname = ''
    for key in kwargs:
        if key not in exclude:
            
            # if the dict contains a dict, make sure to blow that up too.
            if type(kwargs[key]) is dict:
                kw2 = kwargs[key]
                for k2 in kw2:
                    fname += k2+'='+str(kw2[k2])+'_'
            else:
                    
                fname += key+'='+str(kwargs[key])+'_'
    fname += ftype
    
    return fname


def load_sols(fnames):
    """
    load solutions given list of filenames fnames

    """
    
    data_list = []
    
    for i in range(len(fnames)):
        data = np.loadtxt(fnames[i])
        data_list.append(data)
        
    return data_list


def generate_fnames(obj,model_pars='',coupling_pars=''):
    """
    each 'fnames' is a dict containing lists. for each key corresponding
    to the variables in obj.var_names, the list of names correspond
    to the epsilon  or psi power of approximation.
    
    example:
    g_fname = {'v':['fname for gv0','fname for gv1','fname for gv2'],
               'h':['fname for gh0','fname for gh1','fname for gh2']}
    
    to get the file name for gv1, use
    
    g_fname['v'][1]
    
    the notation is generalizable and helps
    keep the code less cluttered.
    
    
    """    
    c_pars = coupling_pars

    for i in range(obj.N):
        
        val = '{}p{}_sym_{}_N={}.d'
        obj.p[i]['sym_fnames'] = [val.format(obj.dir,i,k,obj.N)
                                  for k in range(obj.miter)]

        val = '{}p{}_lam_{}_N={}.d'
        obj.p[i]['lam_fnames'] = [val.format(obj.dir,i,k,obj.N)
                                  for k in range(obj.miter)]
        
        val = '{}p{}_dat_{}{}{}_NA={}_piter={}_N={}.txt'
        
        obj.p[i]['dat_fnames'] = [val.format(obj.dir,i,k,model_pars,
                                             c_pars,obj.NH,obj.p_iter,
                                             obj.N)
                                  for k in range(obj.miter)]

    for i in range(obj.N):
        val = '{}h{}_sym_{}_N={}.d'
        obj.h[i]['sym_fnames'] = [val.format(obj.dir,i,k,obj.N)
                                  for k in range(obj.miter)]
        
        val = '{}h{}_dat_{}{}{}_NA={}_piter={}_N={}.txt'
        obj.h[i]['dat_fnames'] = [val.format(obj.dir,i,k,model_pars,
                                             c_pars,obj.NH,obj.p_iter,
                                             obj.N)
                                  for k in range(obj.miter)]

        val = '{}h{}_lam_{}_N={}.d'
        obj.h[i]['lam_fnames'] = [val.format(obj.dir,i,k,obj.N)
                                  for k in range(obj.miter)]
        val = '{}h{}_lam_{}_N={}_max_n={}.d'
        obj.h[i]['lam_fnames_fourier'] = [val.format(obj.dir,i,k,obj.N,obj.max_n)
                                        for k in range(obj.miter)]



    # messy but keep it for now
    for key in obj.system.var_names:
        for i in range(obj.N):
            obj.k[i]['sym_fnames_'+key] = []

            for j in range(obj.N):
                val = '{}k{}{}{}_sym_{}.d'
                obj.k[i]['sym_fnames_'+key].append([val.format(obj.dir,key,i,j,k)
                                                    for k in range(obj.miter)])
                
    for i in range(obj.N):
        obj.c[i]['sym_fname'] = []
        for j in range(obj.N):
            val = '{}c{}{}_N={}.d'
            obj.c[i]['sym_fname'].append(val.format(obj.dir,i,j,obj.N))

                
def plot(obj,option='g1'):
    
    # check if option of the form 'g'+'int'
    if (option[0] == 'g') and (option[1:].isnumeric()):
        k = int(option[1:])
        
        fig, axs = plt.subplots(nrows=obj.dim,ncols=1,figsize=(4,4))
        
        for i,ax in enumerate(axs):
            key = obj.var_names[i]
            ax.plot(obj.tlc,obj.g['dat'][k][:,i],label=key)
            ax.legend()
            
        axs[0].set_title('g'+str(k))
        plt.tight_layout()
        #ax.plot(obj.tlc,obj.z_data[k][:,1],'zy true')
        #ax.legend()
        
    if (option[0] == 'z') and (option[1:].isnumeric()):
        k = int(option[1:])
        
        fig, axs = plt.subplots(nrows=obj.dim,ncols=1,figsize=(4,4))
        
        for i,ax in enumerate(axs):
            key = obj.var_names[i]
            ax.plot(obj.tlc,obj.z['dat'][k][:,i],label=key)
            ax.legend()
        
        axs[0].set_title('z'+str(k))
        #ax.set_title(option)
        plt.tight_layout()
        
        #print('z init',k,obj.z['dat_'][k][0,:])

    # check if option of the form 'g'+'int'
    if (option[0] == 'i') and (option[1:].isnumeric()):
        k = int(option[1:])
        
        fig, axs = plt.subplots(nrows=obj.dim,ncols=1,figsize=(4,4))
        
        for i,ax in enumerate(axs):
            key = obj.var_names[i]
            ax.plot(obj.tlc,obj.i['dat'][k][:,i],label=key)
            ax.legend()
        
        axs[0].set_title('i'+str(k))
        #ax.set_title(option)
        plt.tight_layout()
    
    if (option[:2] == 'pA') and (option[2:].isnumeric()):
        k = int(option[2:])
        
        if k > 0:
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            #ax.plot_surface(obj.A_mg,obj.B_mg,
            #                obj.pA_callable[k](obj.A_array,obj.B_array),
            #                cmap='viridis')
            
            ax.plot_surface(obj.A_mg,obj.B_mg,
                            obj.pA['dat'][k],
                            cmap='viridis')
            
            ax.set_title(option)
            #plt.show(block=True)
            #plt.close()
    
    
    if option == 'surface_z':
        fig = plt.figure(figsize=(6,6))
        ax = fig.gca(projection='3d')
        
        # Make data.
        th = np.arange(0,1, .01)
        psi = np.arange(-1, 1, .01)
        
        th, psi = np.meshgrid(th, psi)
        
        Z = 0
        for i in range(obj.miter):
            Z += psi**i*obj.zx_callable[i](th)
        #Z = obj.zx_callable[0](th)+psi*obj.zx_callable[1](th)

        # Plot the surface.
        ax.plot_surface(th, psi, Z,cmap='viridis')
        
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\psi$')
        ax.set_zlabel(r'$\mathcal{Z}$')
        
        
        ax.view_init(15,-45)
        ax.dist = 12
        
        plt.tight_layout()


    if option == 'surface_i':
        # # Make data.
        th = np.arange(0,1,.01)
        psi = np.arange(-1,1,.01)

        fig = plt.figure(figsize=(6,6))
        ax = fig.gca(projection='3d')
        
        th, psi = np.meshgrid(th, psi)
        
        ifinal = 0
        
        for i in range(obj.miter):
            ifinal += obj.ix_callable[i](th)*psi**i
       
        # Plot the surface.
        ax.plot_surface(th, psi, ifinal,cmap='viridis')
        
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\psi$')
        ax.set_zlabel(r'$\mathcal{I}$')
        
        
        ax.view_init(15,-45)
        ax.dist = 12
        #ax.set_zlim(-10,10)
        
        if False:
       
            
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            
            ix1 = 0
            ix2 = 0
            ix3 = 0
            
            for i in range(len(obj.ix_callable)):
                ix1 += obj.ix_callable[i](th[0])*psi**i
                ix2 += obj.ix_callable[i](th[25])*psi**i
                ix3 += obj.ix_callable[i](th[50])*psi**i

            ax.plot(psi,ix1)
            ax.plot(psi,ix2)
            ax.plot(psi,ix3)
           
        plt.tight_layout()
        
        
    if (option[:4] == 'hodd') and (option[4:].isnumeric()):
        k = int(option[4:])
        
        #A_array,dxA = np.linspace(0,self.T,self.NA[k],retstep=True)
        B_array,dxB = np.linspace(0,obj.T,obj.NB[k],retstep=True)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(B_array,obj.hodd['dat'][k])
        ax.set_xlabel('theta')
        ax.set_ylabel('H')
        NA = obj.NH
        NB = obj.NB
        Ns = obj.Ns
        #print(obj.Ns)
        #p_iter = obj.p_iter
        title = ('All H of order '+str(k)
                 +', NA='+str(NA[k])
                 +', NB='+str(NB[k])
                 +', Ns='+str(Ns[k]))
                 #+', piter='+str(p_iter[k]))
        ax.set_title(title)
        plt.tight_layout()



