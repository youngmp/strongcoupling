"""
Utility library functions
"""
from . import rhs

import time
import os
import dill
import sys

import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt

from scipy.optimize import bisect

from copy import deepcopy

rhs_avg_ld = rhs.rhs_avg_1d
rhs_avg = rhs.rhs_avg_2d

#from scipy.interpolate import interp1d
from sympy.physics.quantum import TensorProduct as kp
#from sympy.utilities.lambdify import lambdify, implemented_function

from scipy.integrate import solve_ivp
kw_bif = {'method':'LSODA','dense_output':True,'rtol':1e-9,'atol':1e-9}

def get_phase(t,sol_arr,skipn,system1):
    """
    Get the brute-force phase estimate over time
    t: time array
    sol_arr: solution array (must include all dimensions of 
    one and only one oscillator)
    skipn: number of points to skip
    system1: object of response functions
    """

    phase1 = np.zeros(len(t[::skipn]))
    for i in range(len(t[::skipn])):
        d1 = np.linalg.norm(sol_arr[::skipn][i,:]-system1.lc['dat'],axis=1)
        idx1 = np.argmin(d1)
        
        phase1[i] = idx1/len(system1.lc['dat'])
        
    return t[::skipn],2*np.pi*phase1


def freq_est(t,y,transient=.5,width=10,prominence=.15,return_idxs=False):
    """ 
    Estimate the oscillator frequency
    For use only with the frequency ratio plots
    """
    peak_idxs = sp.signal.find_peaks(y,width=width,prominence=prominence)[0]
    peak_idxs = peak_idxs[int(len(peak_idxs)*transient):]
    freq = 2*np.pi/np.mean(np.diff(t[peak_idxs]))
    
    if return_idxs:
        return freq,peak_idxs
    else:
        return freq

def pl_exist_1d(eps,del1,a,th_init=0,return_data=False):
    """
    
    """
    sys1 = a.system1
    
    if eps == 0:
        return -1
    err = 1
    
    th_temp = np.linspace(0, 2*np.pi, 200)
    rhs = rhs_avg_ld(0,th_temp,a,eps,del1)

    if return_data:
        return th_temp,rhs
    
    if np.abs(np.sum(np.sign(rhs))) < len(rhs):
        return 1
    else:
        return -1
    
def get_tongue_1d(del_list,a,deps=.002,max_eps=.3,min_eps=0):
    """
    Get the Arnold tongue for the low-dim reduced model
    (where the isostable coordinate was eliminated)
    """
    ve_exist = np.zeros(len(del_list))
    
    for i in range(len(del_list)):
        print(np.round((i+1)/len(del_list),2),'    ',end='\r')

        if np.isnan(ve_exist[i-1]):
            eps = 0
        else:
            eps = max(ve_exist[i-1] - 2*deps,0)
        while not(pl_exist_ld(eps,del_list[i],a)+1)\
        and eps <= max_eps:
            eps += deps
            #print('existloop',eps)
        if eps >= max_eps:
            ve_exist[i] = np.nan
        else:
            deps2 = deps
            flag = False
            while not(flag) and deps2 < .2:
                #print('while loop',deps2)
                try:
                    out = bisect(pl_exist_ld,0,eps+deps2,args=(del_list[i],a))
                    flag = True
                except ValueError:
                    deps2 += .001
            if flag:
                ve_exist[i] = out
            else:
                ve_exist[i] = np.nan
    print('')
    return del_list,ve_exist

def is_stable(J):
    u,v = np.linalg.eig(J)
    print('eigs',u)
    print('')
    # if all real parts are negative, return true
    if np.sum(np.real(u)<0) == len(u):
        return True
    else:
        return False

def jac_2d(y,a,eps,del1,h=.01):
    """
    Jacobian for 2d averaged system (forcing only)
    """
    print('y',y,'eps',eps,'del1',del1)
    dx = np.array([h,0])
    dy = np.array([0,h])

    args = (a,eps,del1)
    
    c1 = (rhs.rhs_avg_2d(0,y+dx,*args)-rhs.rhs_avg_2d(0,y,*args))/h
    c2 = (rhs.rhs_avg_2d(0,y+dy,*args)-rhs.rhs_avg_2d(0,y,*args))/h

    return np.array([c1,c2])


def get_dy(rhs,Y,args_temp,eps_pert=1e-4,eps_pert_time=1e-4,sign=1,
           return_sol=False):
    J = np.zeros([len(Y),len(Y)])

    for jj in range(len(Y)-1):
        pert = np.zeros(len(Y)-1);pert[jj] = eps_pert

        t_span = [0,Y[-1]]
        solp = solve_ivp(y0=Y[:-1]+pert,t_span=t_span,**args_temp)
        solm = solve_ivp(y0=Y[:-1]-pert,t_span=t_span,**args_temp)

        # Jacobian
        J[:-1,jj] = (solp.y.T[-1,:]-solm.y.T[-1,:])/(2*eps_pert)

    J[:-1,:-1] = J[:-1,:-1] - np.eye(len(Y)-1)

    t_spanp = [0,Y[-1]+eps_pert_time]
    t_spanm = [0,Y[-1]-eps_pert_time]
    solpt = solve_ivp(y0=Y[:-1],t_span=t_spanp,**args_temp)
    solmt = solve_ivp(y0=Y[:-1],t_span=t_spanm,**args_temp)

    J[:-1,-1] = (solpt.y.T[-1,:]-solmt.y.T[-1,:])/(2*eps_pert_time)
    J[-1,-1] = 0
    J[-1,:-1] = rhs(0,Y[:-1],*args_temp['args'])

    solb = solve_ivp(y0=Y[:-1],t_span=t_span,**args_temp)

    b = np.array(list(Y[:-1]-solb.y.T[-1,:])+[0])
    dy = np.linalg.solve(J,b)

    if return_sol:
        return dy,solb
    else:
        return dy

def pl_exist_2d(eps,del1,a,th_init=0,return_data=False,
             pmin=-.25,pmax=.25):
    """
    Check existence of phase-locked solutions in forced system
    given planar reduction.
    """
    system1 = a.system1

    if eps == 0:
        return -1
    # every intersection point must be within eps of a point on the other
    # contour path
    err = .1
    
    # cluster together intersection points so that the
    # original points in each flat cluster have a
    # cophenetic_distance < cluster_size
    # from stackoverflow
    cluster_size = 100
    
    th_temp = np.linspace(0, 2*np.pi, 1000)
    ps_temp = np.linspace(pmin, pmax, 1000)
    
    TH,PS = np.meshgrid(th_temp,ps_temp)

    Z1,Z2 = rhs_avg(0,[TH,PS],a,eps,del1)
    fig_temp,axs_temp = plt.subplots()

    contour1 = axs_temp.contour(TH,PS,Z1,levels=[0],linewidths=.5,colors='k')
    contour2 = axs_temp.contour(TH,PS,Z2,levels=[0],linewidths=.5,colors='b')

    plt.close(fig_temp)
    
    if return_data:
        points1 = contour_points(contour1)
        points2 = contour_points(contour2)

        if isinstance(points1, int) or isinstance(points2, int):
            data = np.array([[np.nan,np.nan]])

        else:
            intersection_points = intersection(points1, points2, err)

            if len(intersection_points) == 0:
                data = np.array([[np.nan,np.nan]])
            
            else:
                data = cluster(intersection_points,cluster_size)

        return contour1,contour2,data
    else:
        return contour1,contour2

def get_tongue_2d(del_list,a,deps=.002,max_eps=.3,min_eps=0):

    ve_exist = np.zeros(len(del_list))
    
    for i in range(len(del_list)):
        print(np.round((i+1)/len(del_list),2),'    ',end='\r')

        if np.isnan(ve_exist[i-1]):
            eps = 0
        else:
            eps = max(ve_exist[i-1] - 2*deps,0)
        while not(pl_exist(eps,del_list[i],a)+1)\
        and eps <= max_eps:
            eps += deps
            #print('existloop',eps)
        if eps >= max_eps:
            ve_exist[i] = np.nan
        else:
            deps2 = deps
            flag = False
            while not(flag) and deps2 < .2:
                #print('while loop',deps2)
                try:
                    out = bisect(pl_exist,0,eps+deps2,args=(del_list[i],a))
                    flag = True
                except ValueError:
                    deps2 += .001
            if flag:
                ve_exist[i] = out
            else:
                ve_exist[i] = np.nan
    print('')
    return del_list,ve_exist

import collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier


def intersection(points1, points2, eps):
    tree = spatial.KDTree(points1)
    distances, indices = tree.query(points2, k=1, distance_upper_bound=eps)
    intersection_points = tree.data[indices[np.isfinite(distances)]]
    return intersection_points


def cluster(points, cluster_size):
    dists = dist.pdist(points, metric='sqeuclidean')
    linkage_matrix = hier.linkage(dists, 'average')
    groups = hier.fcluster(linkage_matrix, cluster_size, criterion='distance')
    return np.array([points[cluster].mean(axis=0)
                     for cluster in clusterlists(groups)])


def contour_points(contour, steps=1):
    for linecol in contour.collections:
        
        if len(linecol.get_paths()) == 0:
            return 0
    
    return np.row_stack([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])

def clusterlists(T):
    '''
    http://stackoverflow.com/a/2913071/190597 (denis)
    T = [2, 1, 1, 1, 2, 2, 2, 2, 2, 1]
    Returns [[0, 4, 5, 6, 7, 8], [1, 2, 3, 9]]
    '''
    groups = collections.defaultdict(list)
    for i, elt in enumerate(T):
        groups[elt].append(i)
    return sorted(groups.values(), key=len, reverse=True)


def get_period(sol,idx:int=0,prominence:float=.25)->float:
    """
    sol: solution object from solve_ivp
    idx: index of desired variable

    TODO: need to make sure that subsequent peaks are
    actually after 1 full period.
    """
    
    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],
                                     prominence=prominence)[0]
    maxidx = peak_idxs[-2]    
    maxidx_prev = peak_idxs[-3]

    #t_prev_est = (sol.t[maxidx]-init[-1])
    #t_prev_idx = np.argmin(np.abs(t_prev_est-sol.t))
    #maxidx_est = np.argmin(np.abs(peak_idxs-t_prev_idx))
    #maxidx_prev = peak_idxs[maxidx_est]

    def sol_min(t):
        return -sol.sol(t)[idx]

    # get stronger estimates of max values
    # use optimization on ode solution
    pad1lo = (sol.t[maxidx]-sol.t[maxidx-1])/2
    pad1hi = (sol.t[maxidx+1]-sol.t[maxidx])/2
    bounds1 = [sol.t[maxidx]-pad1lo,sol.t[maxidx]+pad1hi]
    res1 = sp.optimize.minimize_scalar(sol_min,bounds=bounds1)

    pad2lo = (sol.t[maxidx_prev]-sol.t[maxidx_prev-1])/2
    pad2hi = (sol.t[maxidx_prev+1]-sol.t[maxidx_prev])/2
    bounds2 = [sol.t[maxidx_prev]-pad2lo,sol.t[maxidx_prev]+pad2hi]
    res2 = sp.optimize.minimize_scalar(sol_min,bounds=bounds2)

    T_init = res1.x - res2.x
    
    return T_init,res1.x

def get_initial_phase_diff_c(phi0,a,eps,del1,max_time=3000):
    
    init1 = list(a.system1.lc['dat'][int(phi0*a.system1.TN/(2*np.pi)),:])
    init2 = list(a.system2.lc['dat'][0,:])
    init = np.array(init1+init2+[2*np.pi])
    
    # run for a while and get period
    sol = solve_ivp(_full,[0,max_time],init[:-1],
                    args=(a,eps,del1),**kw_bif)
    
    # get period estimate and time at which period was estimated.
    period_est,time_est = get_period(sol)
    
    # get solution values at time_est.
    init = list(sol.sol(time_est))
    init = np.array(init+[period_est])
    
    return init

def get_phase_diff_f(rhs,phi0,a,eps,del1,u_sign=-1,
                     max_time=3000):
    """
    find initial condition on limit cycle. For forced systems only.
    does not extract phase, only state variables and period.

    u_sign: simple way to make sure that phase zero starts at appropriate
    place for certain forcing functions.
    """
    
    init1 = list(a.system1.lc['dat'][int(phi0*a.system1.TN/(2*np.pi)),:])
    init = np.array(init1+[2*np.pi])
    
    # run for a while and get period
    t = np.arange(0,max_time,.01)
    sol = solve_ivp(rhs,[0,max_time],init[:-1],
                    args=(a,eps,del1),t_eval=t,
                    **kw_bif)

    u_temp = a.system1.forcing_fn(t*(a._m[1]+del1))
    
    # get period estimate
    period_est,time_est = get_period(sol)

    peak_idxs1 = sp.signal.find_peaks(sol.y.T[:,0],prominence=.25)[0]
    peak_idxs2 = sp.signal.find_peaks(-u_temp)[0]

    if len(peak_idxs1) > len(peak_idxs2):
        peak_idxs1 = peak_idxs1[:len(peak_idxs2)]
    else:
        peak_idxs2 = peak_idxs2[:len(peak_idxs1)]

    #tp,fp = get_phase(t,sol.y.T,skipn=100,system1=a.system1)
    
    #fp2 = (t[peak_idxs2]-a.om*t[peak_idxs1])[-1]
    #phi = np.mod(2*np.pi*fp2/(period_est),2*np.pi)


    # get initial condition at zero phase of forcing function
    def u_min(t):
        return a.system1.forcing_fn(t*(a._m[1]+del1))

    ress = []
    for shift in range(a._m[1]):
        bounds1 = [time_est-period_est*(1+2*shift)*a._n[1]/2/a._m[1],
                   time_est+period_est*(1-2*shift)*a._n[1]/2/a._m[1]]
        res1 = sp.optimize.minimize_scalar(u_min,bounds=bounds1)
        ress.append(res1.x)

    ress = np.asarray(ress)

    # diff taken in time not phase, so order is reversed.
    phis = np.mod(2*np.pi*(ress - time_est)/(period_est),2*np.pi)
    print('times full vs force',time_est,res1.x,phis,'eps',eps,'per',period_est)

    
    if True:

        fig,axs = plt.subplots(3,1,figsize=(4,2))

        axs[0].scatter(t[peak_idxs1],(t[peak_idxs2]-t[peak_idxs1]*a.om),
                       s=5,color='gray',alpha=.5,label='Full')

        axs[1].plot(sol.t,sol.y[0])
        axs[1].plot(t,u_temp)

        axs[1].scatter(t[peak_idxs1],sol.y[0,peak_idxs1])
        axs[1].scatter(t[peak_idxs2],u_temp[peak_idxs2])

        axs[2].plot(sol.t,sol.y[0])
        axs[2].plot(t,u_temp)
        axs[2].set_xlim(time_est-10,time_est+10)

        #axs[0].set_xlim(t[peak_idxs1][-1]-50,t[peak_idxs1][-1])
        axs[1].set_xlim(t[peak_idxs1][-1]-50,t[peak_idxs1][-1])
        
        axs[0].set_ylim(0,2*np.pi)
        
        #print('times full vs force',t[peak_idxs1][-1],t[peak_idxs2][-1],
        #      t[peak_idxs2][-1]-t[peak_idxs1][-1]*a.om)
        
        plt.savefig('figs_temp/phi0_eps='+str(eps)+'del='+str(del1)+'.png')
        plt.close()

    
    return np.asarray(phis)


def run_bif1d_f(rhs,Y,a,del1,eps_init,eps_final,deps,
                maxiter:int=100,tol:float=1e-10,
                u_sign=-1,mult=1):
    """
    1d bifurcation for forcing function
    """

    print('u_sign -- must be set manually',u_sign)

    Y[-1] *= mult

    print('integrating bif. over {} periods'.format(mult))
    
    Y_init = deepcopy(Y)
    dy_init = np.ones(len(Y))/5
    eps_range = np.arange(eps_init,eps_final,deps)
    phase_diffs = np.zeros([len(eps_range),2])

    osc2_idx = len(a.system1.var_names)
    
    for i,eps in enumerate(eps_range):
        
        dy = deepcopy(dy_init)
        
        counter = 0
        print('eps iter',i,eps,end='                    \n')
        
        while np.linalg.norm(dy) > tol and\
        counter <= maxiter and\
        np.linalg.norm(dy)<1:
            
            args_temp = {'fun':rhs,'args':(a,eps,del1),**kw_bif}
            dy,solb = get_dy(rhs,Y,args_temp,return_sol=True,
                             eps_pert=1e-2,eps_pert_time=1e-2)
            
            if False:
                solt = solve_ivp(t_span=[0,50],y0=Y[:-1],**args_temp)
                ut = a.system1.forcing_fn(solt.t*(a._m[1]+args_temp['args'][2]))
                a = args_temp['args'][0]
                u = a.system1.forcing_fn(solb.t*(a._m[1]+args_temp['args'][2]))
                fig,axs = plt.subplots()
                axs.plot(solb.t,solb.y[0])
                axs.plot(solt.t,solt.y[0])
                axs.plot(solb.t,u)
                axs.plot(solt.t,ut)
                axs.set_title('eps={}, del1={}'.format(args_temp['args'][1],args_temp['args'][2]))
                plt.savefig('figs_temp/newton_f_eps={}_del1={}_{}.png'.format(args_temp['args'][1],args_temp['args'][2],counter))
                plt.close()

            Y += dy
            print('dy',dy,'Y',Y)
            
            str1 = 'iter={}, LC rel. err ={:.2e}'
            end = '                                       \r'
            printed = str1.format(counter,np.linalg.norm(dy))
            print(printed,end=end)
            counter += 1
            
            time.sleep(.2)
    
        if (counter >= maxiter) or (np.linalg.norm(dy) >= 1):
            phase_diffs[i,:] = np.nan
            Y = Y_init
            dy = np.ones(len(Y))/5
        
        else:
            solf = solve_ivp(y0=Y[:-1],t_span=[0,Y[-1]],**args_temp)

            idx1 = np.argmax(solf.y.T[:,0])
            u = a.system1.forcing_fn(solf.t*(a._m[1]+del1))
            idx2 = sp.signal.find_peaks(u_sign*u)[0]

            if True:

                fig,axs = plt.subplots()
                axs.plot(solf.t,solf.y[0])
                axs.plot(solf.t,u)

                axs.scatter(solf.t[idx1],solf.y[0,idx1])
                axs.scatter(solf.t[idx2],u[idx2])
                
                axs.set_title('eps={}, del1={}'.format(eps,del1))
                plt.savefig('figs_temp/{}.png'.format(counter))
                plt.close()


            if len(idx2) == 0:
                Y = Y_init
                dy = dy_init
                t_diff = phase_diffs[i,:] = np.nan
                
            else:
                t_diff = solf.t[idx2]-solf.t[idx1]
                period = Y[-1]/mult
                #print('ts',solf.t[idx1],solf.t[idx2])
                print('t_diff',t_diff)
                phase_diffs[i,:] = 2*np.pi*np.mod(t_diff,period)/period
        
        
    return eps_range,phase_diffs
