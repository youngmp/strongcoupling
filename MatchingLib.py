# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:49:28 2020

@author: youngmin

library functions
"""

import time
import os
import dill

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sympy.physics.quantum import TensorProduct as kp
from sympy.utilities.lambdify import lambdify, implemented_function

from scipy.integrate import odeint, solve_ivp


def vec(a):
    """
    vec array operator. stack columns.
    
    reshape command stacks rows. so transpose then reshape to stack columns
    
    https://stackoverflow.com/questions/55444777/numpy-array-stack-multiple-columns-into-one-using-reshape
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
        #print('############ df, i,k',np.shape(df),i,k,df)  
        #print('i,df',i,df)
    
    #print(np.shape(df))
    #print('############ df, k',np.shape(df),k,df)    
    return df
    

def rhs(z,t,f):
    """

    Parameters
    ----------
    z : array
        position
    t : scalar
        time.
    f : sympy lambdified symbol R^n->R^n
        right-hand side of ODE.

    Returns
    -------
    array (based on lambdified function f)
        derivative output.

    """
    x = z[0]
    y = z[1]
    
    return f(x,y)


def monodromy(t,z,jacLC):
    """
    calculate right-hand side of system
    \dot \Phi = J\Phi, \Phi(0)=I
    \Phi is a matrix solution
    
    jacLC is the jacobian evaluated along the limit cycle
    """
    #x=z[0];y=z[1]
    
    #\Phi
    
    z = np.reshape(z,(2,2))
    
    dy = np.dot(jacLC(t),z)
    
    
    return np.reshape(dy,4)


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


def files_exist(*fnames):

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


def run_newton(obj,fn,init,hetx_lam,hety_lam,k,
               max_iter=200,rel_tol=1e-12,rel_err=10,eps=1e-1,
               backwards=True):
        
    
    


    if backwards:
        tLC = -obj.tLC
    else:
        tLC = obj.tLC
    # run newton's method
    counter = 0
    while (rel_err > rel_tol) and (counter < max_iter):
        
        dx, t,sol = get_newton_jac(obj,fn,tLC,init,
                                   hetx_lam,hety_lam,k,eps=eps,
                                   return_sol=True)
        
        rel_err = np.amax(np.abs(sol[-1,:]-sol[0,:]))/np.amax(np.abs(sol))
        
        init += dx
        counter += 1
        
        if False:
            #print('dx',dx)
            #print(obj.tLC)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sol)
            plt.show(block=True)
            time.sleep(2)
            plt.close()
        
            print(rel_err)
        #print(counter,np.amax(np.abs(dx)),dx,rel_tol,k)
        if counter == max_iter-1:
            print('WARNING: max iter reached in newton call')
            
    return init


def get_newton_jac(obj,fn,tLC,init,hetx_lam,hety_lam,k,eps=0.1,
                   return_sol=False):
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
    n = len(init)
    
    J = np.zeros((n,n))
    
    sol = solve_ivp(fn,[0,tLC[-1]],init,args=(hetx_lam,hety_lam,k),
                    method=obj.method,dense_output=True,
                    t_eval=tLC,
                    rtol=obj.rtol,atol=obj.atol)
            
    sol_unpert = sol.y.T
    t_unpert = sol.t
    #sol_unpert = odeint(f,init,tLC,args=(hetx_lam,hety_lam,k))
    
    
    for p in range(len(init)):
        
        pert = np.zeros(len(init))
        pert[p] = eps
        pert_init = init + pert
        
        sol = solve_ivp(fn,[0,tLC[-1]],pert_init,args=(hetx_lam,hety_lam,k),
                        method=obj.method,dense_output=True,
                        t_eval=tLC,
                        rtol=obj.rtol,atol=obj.atol)
            
        sol_pert = sol.y.T
        
    
        
        J[:,p] = (sol_pert[-1,:] - sol_unpert[-1,:])/eps
    
        if False:
            #print(obj.tLC)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sol_unpert,color='blue')
            ax.plot(sol_pert)
            ax.set_title('pert'+str(p))
            plt.show(block=True)
            time.sleep(2)
            plt.close()
            print(J)
    
        
    mydiff = sol_unpert[-1,:] - sol_unpert[0,:]
    
    if False:
        arr = np.linspace(0,2*np.pi,100)
        #print('di',self.di(.7,[.4,.7],hetx_lam,hety_lam,0))
        print(hetx_lam(arr),hety_lam(arr))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fn(arr,[.8,.2],hetx_lam,hety_lam,0))
        ax.set_title('di')
        plt.show(block=True)
        time.sleep(2)
        plt.close()
    
    dx = -np.dot(np.linalg.inv(J-np.identity(n)), mydiff)
    #print(dx,'\t',mydiff,'\t',J[:,0],J[:,1])
    
    if return_sol:
        return dx, t_unpert,sol_unpert
    else:
        return dx


    """
    J = np.zeros((2,2))
    sol_unpert = odeint(f,init,tLC,args=(hetx_lam,hety_lam,k))
    
    for p in range(len(init)):
        
        pert = np.zeros(len(init))
        pert[p] = eps
        pert_init = init + pert
    
        sol_pert = odeint(f,pert_init,tLC,args=(hetx_lam,hety_lam,k))
        
        J[:,p] = (sol_pert[-1,:] - sol_unpert[-1,:])/eps
        
    mydiff = sol_unpert[-1,:] - sol_unpert[0,:]
    
    dx = -np.dot(np.linalg.inv(J-np.identity(2)), mydiff)
    #print(dx,'\t',mydiff,'\t',J[:,0],J[:,1])
    init += dx
    """


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




def plot(obj,option='g1'):
    
    # check if option of the form 'g'+'int'
    if (option[0] == 'g') and (option[1:].isnumeric()):
        k = int(option[1:])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(obj.tLC,obj.g_data[k][:,0],label='gx ode')
        ax.plot(obj.tLC,obj.g_data[k][:,1],label='gy ode')
        ax.set_title(option)
        
        if k == 1:
            true_gx = obj.q_val*np.sin(obj.q_val*obj.tLC) + np.cos(obj.q_val*obj.tLC)
            true_gy = np.sin(obj.q_val*obj.tLC) - obj.q_val*np.cos(obj.q_val*obj.tLC)
            
            ax.plot(obj.tLC,true_gx,label='gx true')
            ax.plot(obj.tLC,true_gy,label='gy true')
            
            #ax.plot(obj.tLC,obj.z_data[k][:,1],'zy true')
        ax.legend()
        print('g init',k,obj.g_data[k][0,:])
        
    if (option[0] == 'z') and (option[1:].isnumeric()):
        k = int(option[1:])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(obj.tLC,obj.z_data[k][:,0],label='zx ode')
        ax.plot(obj.tLC,obj.z_data[k][:,1],label='zy ode')
        ax.set_title(option)
        
        if k == 0:
            zx_true = np.cos(obj.q_val*obj.tLC)-np.sin(obj.q_val*obj.tLC)/obj.q_val
            zy_true = np.sin(obj.q_val*obj.tLC)+np.cos(obj.q_val*obj.tLC)/obj.q_val
            ax.plot(obj.tLC,zx_true,label='zx true')
            ax.plot(obj.tLC,zy_true,label='zy true')
            #ax.plot(obj.tLC,obj.z_data[k][:,1],'zy true')
        ax.legend()
        
        print('z init',k,obj.z_data[k][0,:])

    # check if option of the form 'g'+'int'
    if (option[0] == 'i') and (option[1:].isnumeric()):
        k = int(option[1:])
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(obj.tLC,obj.i_data[k][:,0],label='ix ode')
        ax.plot(obj.tLC,obj.i_data[k][:,1],label='iy ode')
        ax.set_title(option)
        
        if k == 0:
            ax.plot(obj.tLC,np.cos(obj.q_val*obj.tLC),label='ix true')
            ax.plot(obj.tLC,np.sin(obj.q_val*obj.tLC),label='iy true')
            #ax.plot(obj.tLC,obj.z_data[k][:,1],'zy true')
            
        ax.legend()
        print('i init',k,obj.i_data[k][0,:])
    
    if (option[:2] == 'pA') and (option[2:].isnumeric()):
        k = int(option[2:])
        
        if k > 0:
            
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            #ax.plot_surface(obj.A_mg,obj.B_mg,
            #                obj.pA_callable[k](obj.A_array,obj.B_array),
            #                cmap='viridis')
            
            ax.plot_surface(obj.A_mg,obj.B_mg,
                            obj.pA_data[k],
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
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(obj.B_array,obj.h_odd_data[k])
        ax.set_xlabel('theta')
        ax.set_ylabel('H')
        ax.set_title('All H of order '+str(k))
        plt.tight_layout()
