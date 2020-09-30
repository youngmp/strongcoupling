"""
Generate figures for strong coupling paper
"""

#from decimal import Decimal
#from matplotlib.collections import PatchCollection
import os
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)
<<<<<<< Updated upstream

=======
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
>>>>>>> Stashed changes
from matplotlib.legend_handler import HandlerBase
from scipy.optimize import brentq

import CGL
import Thalamic

from StrongCoupling import StrongCoupling


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
preamble = r'\usepackage{amsmath} \usepackage{siunitx}'
matplotlib.rcParams['text.latex.preamble'] = preamble

#import matplotlib.patches as patches
#from matplotlib.patches import FancyArrowPatch
#from mpl_toolkits.mplot3d import proj3d
#from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)

#from lib import collect_disjoint_branches

#import matplotlib as mpl

# font size
size = 12

exp = np.exp
pi = np.pi

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           linestyle='--', color=orig_handle[0])
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], 
                           color=orig_handle[0])
        return [l1, l2]

def return_slope_sign(eps,hterms,state='sync',order=8):
    # return slope at sync or antiphase given eps.
    
    h = 0
    # construct h function
    for i in range(order):
        
        # convert h data to function
        #hfn = 
        
        h += eps**(i+1)*hterms[i]
    
    if state == 'sync':
        diff = h[1] - h[0]
    
    elif state == 'anti':
        diff = h[int(len(h)/2)] - h[int(len(h)/2)-1]
        
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(h)
        ax.set_title('eps='+str(eps))
        plt.show(block=True)
        
    #print(diff)
        
    return diff


def h(order,hs,eps):

    # construct h function
    h = 0
    for i in range(order):
        h += eps**(i+1)*hs[i]
    return h


def cgl_h():
    
    fig, axs = plt.subplots(nrows=2,ncols=3,
                            figsize=(7,3.5))
    
    d_center = 1
    d_vals = np.linspace(d_center-1,d_center+1,100)[10:][::2]
    d1 = d_vals[6]
    d2 = d_vals[3]

    order_list = [2,4,10]
    
    var_names = ['x','y']
    
    pardict = {'q_val':1,
               'eps_val':0,
               'd_val':1}
    
    kwargs = {'recompute_LC':False,
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
              'trunc_order':9,
              'dir':'cgl_dat/',
              'NA':501,
              'NB':501,
              'p_iter':25,
              'TN':2001,
              'rtol':1e-7,
              'atol':1e-7,
              'rel_tol':1e-6,
              'method':'LSODA',
              'load_all':False}
    
    T_init = 2*np.pi
    LC_init = np.array([1,0,T_init])

    kwargs['d_val'] = d1
    a1 = StrongCoupling(CGL.rhs,CGL.coupling,LC_init,
                           var_names,pardict,**kwargs)

    kwargs['d_val'] = d2
    a2 = StrongCoupling(CGL.rhs,CGL.coupling,LC_init,
                           var_names,pardict,**kwargs)

    a1.load_h()
    a2.load_h()

    eps1 = .3
    eps2 = -.7

    label1 = [r'\textbf{A}'+r' $d=%.1f$, $\varepsilon=%.1f$'
                       % (a1.d_val, eps1),r'\textbf{B}',r'\textbf{C}']
    label2 = [r'\textbf{D}'+r' $d=%.1f$, $\varepsilon=%.1f$'
                       % (a2.d_val, eps2),r'\textbf{E}',r'\textbf{F}']
    
    xs = [0.35,0,0]
    
    axs[0,0].set_ylabel(r'$-2\mathcal{H}_\text{odd}(\phi)$')
    axs[1,0].set_ylabel(r'$-2\mathcal{H}_\text{odd}(\phi)$')
    
    for i in range(3):
        h1 = h(order_list[i],a1.hodd['dat'],eps1)
        h2 = h(order_list[i],a2.hodd['dat'],eps2)

        x = np.linspace(0,a1.T,len(h1))

        # h functions
        axs[0,i].plot(x,h1,color='k',label='Order '+str(order_list[i]))
        axs[1,i].plot(x,h2,color='k',label='Order '+str(order_list[i]))

        # zero line
        axs[0,i].plot([0,2*np.pi],[0,0],color='gray',lw=.5,zorder=-1)
        axs[1,i].plot([0,2*np.pi],[0,0],color='gray',lw=.5,zorder=-1)
        
        axs[0,i].set_xlim(0,2*np.pi)
        axs[1,i].set_xlim(0,2*np.pi)

        axs[0,i].text(4,np.amin(h1)/2,'Order '+str(order_list[i]))
        axs[1,i].text(4,np.amax(h2)/2,'Order '+str(order_list[i]))

        axs[1,i].set_xlabel(r'$\phi$')
        
        axs[0,i].set_title(label1[i],x=xs[i])
        axs[1,i].set_title(label2[i],x=xs[i])
        #axs[1,i].set_title('d='+str(a2.d_val)
        #                   +'Order '+str(order_list[i]))
        #axs[0,i].legend(fontsize=8)
        #axs[1,i].legend(fontsize=8)
        
        axs[0,i].set_xticks([0,np.pi,2*np.pi])
        axs[1,i].set_xticks([0,np.pi,2*np.pi])
        
        axs[0,i].set_xticklabels(['$0$','$T/2$','$T$'])
        axs[1,i].set_xticklabels(['$0$','$T/2$','$T$'])

        if i == 2:
            axins = inset_axes(axs[1,i], width="40%", height="25%", loc=3)
            #axins.plot([Us[0],Us[-1]],[0,0],color='gray',lw=1)
            upper_idx = 11
            axins.plot(x[:upper_idx],h2[:upper_idx],color='k')
            axins.set_xlim(0,np.amax(x[:upper_idx]))
            axins.set_ylim(np.amin(h2[:upper_idx])-.05,np.amax(h2[:upper_idx])+.05)
            axins.plot([0,2*np.pi],[0,0],color='gray',lw=.5,zorder=-1)

            #axins.scatter(Us[c_idxs],np.zeros(len(Us[c_idxs])),color='k',zorder=6,s=6)    
            mark_inset(axs[1,i], axins, loc1=2, loc2=1, fc="none", ec='0.5',alpha=0.5)

            axins.set_xticks([])
            axins.set_yticks([])
        

    plt.tight_layout()
    return fig


def cgl_2par(recompute_2par=False,recompute_all=False):
    
    fig = plt.figure(figsize=(5,4))
        
    gs = fig.add_gridspec(1, 1)    
    ax = fig.add_subplot(gs[:,:])
    
    # use this array and cut out pieces to avoid lots of recomputing.
    d_center = 1
    d_vals = np.linspace(d_center-1,d_center+1,100)[10:][::2]
    
    var_names = ['x','y']
    
    pardict = {'q_val':1,
               'eps_val':0,
               'd_val':1}
    
    kwargs = {'recompute_LC':False,
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
              'trunc_order':9,
              'dir':'cgl_dat/',
              'NA':501,
              'NB':501,
              'p_iter':25,
              'TN':2001,
              'rtol':1e-7,
              'atol':1e-7,
              'rel_tol':1e-6,
              'method':'LSODA',
              'load_all':True}
    
    T_init = 2*np.pi
    LC_init = np.array([1,0,T_init])
    
    q_val = pardict['q_val']
    #d_val = options['d_val']
    
    fname = ('twopar_cgl'
             +'_order='+str(kwargs['trunc_order'])
             +'.txt')
    
    file_not_found = not(os.path.isfile(fname))
    
    if recompute_2par or file_not_found:
        a = StrongCoupling(CGL.rhs,CGL.coupling,LC_init,
                           var_names,pardict,**kwargs)
        kwargs['load_all'] = False
        
        # first column d, second col sync, third column anti.
        ve_sync_anti = np.zeros((len(d_vals),3))
        ve_sync_anti[:,0] = d_vals
        
        for j,d in enumerate(d_vals):
            print('j,d =',j,d)
            pardict['d_val'] = d
            
            a.__init__(CGL.rhs,CGL.coupling,LC_init,
                       var_names,pardict,**kwargs)
            
            a.load_p_sym()
            a.load_p()
            a.load_h_sym()
            a.load_h()
            #print(a.h_odd_data)
            
            # find zero slope at sync and antiphase in eps
            # break down into positive and negative eps
            
            if d <=1: 
                sgn = 1
            else:
                sgn = -1
                
            # positive eps
            eps_anti = brentq(return_slope_sign,sgn*0.001,sgn*.5,
                              args=(a.hodd['dat'],'anti',
                                    a.trunc_order+1))
            
            # negative eps
            eps_sync = brentq(return_slope_sign,-sgn*0.001,-sgn*1,
                              args=(a.hodd['dat'],'sync',
                                    a.trunc_order+1))
    
            #print('eps0',eps0)
            ve_sync_anti[j,1] = eps_anti
            ve_sync_anti[j,2] = eps_sync
            
        # save data
        np.savetxt(fname,ve_sync_anti)
        
    else:
        # if file exists and not recompute, load data.
        ve_sync_anti = np.loadtxt(fname)


    
    color_true = '.0'
    color_PRL = '#7b3294'
    color_8th = '#008837'

    lw_true = 2
    
    # 10th order correction
    ax.plot(ve_sync_anti[:,0],ve_sync_anti[:,2],color=color_8th,lw=2)
    ax.plot(ve_sync_anti[:,0],ve_sync_anti[:,1],color=color_8th,lw=2,ls='--')
    
    # true cruve stability
    es_true = (d_vals*q_val-1)/(d_vals**2+1)
    ea_true = (1-d_vals*q_val)/(d_vals**2-2*q_val*d_vals + 3)
    
    es_appx = (d_vals*q_val-1)/(d_vals**2*(1+q_val**2))
    ea_appx = (1-d_vals*q_val)/(d_vals**2*(1+q_val**2))
        
    # ground truth
    ax.plot([0,2],[0,0],color=color_true,lw=lw_true)
    ax.plot(d_vals,es_true,color=color_true,lw=lw_true,label='Analytic',zorder=-2)
    ax.plot(d_vals,ea_true,color=color_true,lw=lw_true,zorder=-2,ls='--')

    # 2nd order
    ax.plot(d_vals,es_appx,color=color_PRL,lw=2,label='2nd Order')
    ax.plot(d_vals,ea_appx,color=color_PRL,lw=2,ls='--')

    # fill domains
    e_temp1 = np.append(es_true[d_vals<=1],ea_true[d_vals>1])
    e_temp2 = np.append(ea_true[d_vals<=1],es_true[d_vals>1])

    ec = '0.85'
    ax.fill_between(d_vals,es_true,facecolor='1',edgecolor=ec,zorder=-5,hatch='++',label='10th Order')
    ax.fill_between(d_vals,ea_true,facecolor='1',edgecolor=ec,zorder=-5,hatch='oo')
    ax.fill_between(d_vals,e_temp1,-1,facecolor='1',edgecolor=ec,zorder=-5,hatch='///')
    ax.fill_between(d_vals,e_temp2,1,facecolor='1',edgecolor=ec,zorder=-5,hatch='///')
    
    
    # label regions
    ax.text(1.,0.25,'I',fontsize=16,family='serif',ha='center')
    ax.text(1.,-0.3,'I',fontsize=16,family='serif',ha='center')

    ax.text(0.5,0.02,'II',fontsize=16,family='serif',ha='center')
    ax.text(1.5,-0.08,'II',fontsize=16,family='serif',ha='center')
    
    ax.text(1.5,0.02,'III',fontsize=16,family='serif',ha='center')
    ax.text(0.5,-0.08,'III',fontsize=16,family='serif',ha='center')

    # these coordinates/labels depend on the cgl H function figure
    ax.scatter(0.4,0.3,marker='*',color='k',s=50)
    ax.text(0.425,0.3,'A',va='center',fontsize=12)
    ax.scatter(0.3,-0.7,marker='*',color='k',s=50)
    ax.text(0.325,-0.7,'B',va='center',fontsize=12)
    
    #old_boundary = matplotlib.patches.Rectangle((0.4,-.25),1.4,.5,
    #                                            facecolor="none",
    #                                            ec=(1,0,0,.5),
    #                                            zorder=-3)
    #ax.add_patch(old_boundary)
    
    ax.set_ylim(-.8,.4)
    ax.set_xlim(d_vals[0],d_vals[-1])
    
    ax.set_ylabel(r'Coupling Strength $\varepsilon$')
    ax.set_xlabel(r'Coupling parameter $d$')
    

    #from matplotlib.lines import Line2D
    #custom_lines = [Line2D([0], [0], color=color_true, lw=lw_true),
    #                Line2D([0], [0], color=color_PRL, lw=2),
    #                Line2D([0], [0], color=color_8th, lw=2)]

    #ax.legend(custom_lines, ['True', '2nd Order', '8th Order'])

    ax.legend([(color_true,), (color_PRL,), (color_8th,)],
               ['Analytic', '2nd Order', '10th Order'],
               handler_map={tuple: AnyObjectHandler()})
    
    plt.tight_layout()
    
    return fig

<<<<<<< Updated upstream
def h(order,hs,eps):

    # construct h function
    h = 0
    for i in range(order):
        h += eps**(i+1)*hs[i]
    return h


def cgl_h():
    
    fig, axs = plt.subplots(nrows=2,ncols=3,
                            figsize=(7,3.5))
    
    d_center = 1
    d_vals = np.linspace(d_center-1,d_center+1,100)[10:][::2]
    d1 = d_vals[6]
    d2 = d_vals[3]

    order_list = [2,4,10]

    options = {'recompute_g_sym':False,
               'recompute_g':False,
               'recompute_het_sym':False,
               'recompute_z':False,
               'recompute_i':False,
               'recompute_k_sym':False,
               'recompute_p_sym':False,
               'recompute_p':False,
               'recompute_h_sym':False,
               'recompute_h':False,
               'trunc_order':9,
               'trunc_derviative':2,
               'd_val':1,
               'q_val':1,
               'load_all':False}

    options['d_val'] = d1
    a1 = CGL(**options)

    options['d_val'] = d2
    a2 = CGL(**options)

    a1.load_h()
    a2.load_h()

    eps1 = .3
    eps2 = -.7

    label1 = [r'\textbf{A}'+r' $d=%.1f$, $\varepsilon=%.1f$'
                       % (a1.d_val, eps1),r'\textbf{B}',r'\textbf{C}']
    label2 = [r'\textbf{D}'+r' $d=%.1f$, $\varepsilon=%.1f$'
                       % (a2.d_val, eps2),r'\textbf{E}',r'\textbf{F}']
    
    xs = [0.3,0,0]
    
    axs[0,0].set_ylabel(r'$\mathcal{H}(-\phi)-\mathcal{H}(\phi)$')
    axs[1,0].set_ylabel(r'$\mathcal{H}(-\phi)-\mathcal{H}(\phi)$')
    
    for i in range(3):
        h1 = h(order_list[i],a1.hodd['dat'],eps1)
        h2 = h(order_list[i],a2.hodd['dat'],eps2)

        x = np.linspace(0,a1.T,len(h1))

        # h functions
        axs[0,i].plot(x,h1,color='k',label='Order '+str(order_list[i]))
        axs[1,i].plot(x,h2,color='k',label='Order '+str(order_list[i]))

        # zero line
        axs[0,i].plot([0,2*np.pi],[0,0],color='gray',lw=.5,zorder=-1)
        axs[1,i].plot([0,2*np.pi],[0,0],color='gray',lw=.5,zorder=-1)
        
        axs[0,i].set_xlim(0,2*np.pi)
        axs[1,i].set_xlim(0,2*np.pi)

        axs[0,i].text(4,np.amin(h1)/2,'Order '+str(order_list[i]))
        axs[1,i].text(4,np.amax(h2)/2,'Order '+str(order_list[i]))

        axs[1,i].set_xlabel(r'$\phi$')
        
        axs[0,i].set_title(label1[i],x=xs[i])
        axs[1,i].set_title(label2[i],x=xs[i])
        #axs[1,i].set_title('d='+str(a2.d_val)
        #                   +'Order '+str(order_list[i]))
        #axs[0,i].legend(fontsize=8)
        #axs[1,i].legend(fontsize=8)

        if i == 2:
            axins = inset_axes(axs[1,i], width="40%", height="25%", loc=3)
            #axins.plot([Us[0],Us[-1]],[0,0],color='gray',lw=1)
            upper_idx = 11
            axins.plot(x[:upper_idx],h2[:upper_idx],color='k')
            axins.set_xlim(0,np.amax(x[:upper_idx]))
            axins.set_ylim(np.amin(h2[:upper_idx])-.05,np.amax(h2[:upper_idx])+.05)
            axins.plot([0,2*np.pi],[0,0],color='gray',lw=.5,zorder=-1)

            #axins.scatter(Us[c_idxs],np.zeros(len(Us[c_idxs])),color='k',zorder=6,s=6)    
            mark_inset(axs[1,i], axins, loc1=2, loc2=1, fc="none", ec='0.5',alpha=0.5)

            axins.set_xticks([])
            axins.set_yticks([])
        

    plt.tight_layout()
    return fig
=======
def rhs(t,y,obj,eps):
    """
    get thal rhs out of obj
    """
    
    z = np.zeros(len(y))
    #print('y[:4]',y[:4])
    #print('y[:4]',y[4:])
    y_reverse = np.zeros(8)
    y_reverse[:4] = y[4:]
    y_reverse[4:] = y[:4]
    z[:4] = obj.thal_rhs(t,y[:4]) + eps*obj.thal_coupling(y)
    z[4:] = obj.thal_rhs(t,y[4:]) + eps*obj.thal_coupling(y_reverse)
    #(z,t)
    return z


def thalamic_diffs(ax,a,eps,recompute=False,Tf=0):
    """
    show numerical simulation of phase differences over time
    """
    #T = 10.648268787326938
    
    # simulate series of thalamic models for different initial conditions

    if eps > 0.01 and eps < 0.05:
        T = a.T
        LC = np.loadtxt('thal2_lc_eps=0.dat')
        T2 = T
    elif eps == 0.1 or eps == 0.09:
        LC = np.loadtxt('thal2_lc_eps=0.dat')
        T = LC[-1,0]
        T2 = T
        print('WARNING: using eps=0 period for eps=0.1')
    elif eps == 0.25:
        LC = np.loadtxt('thal2_lc_eps=0.dat')
        
        LC_small = np.loadtxt('thal2_lc_large_eps=.25.dat')
        #LC_large = np.loadtxt('thal2_lc_small_eps=.25.dat')
        
        #T = LC_small[-1,0]
        #T2 = 8.43
        
        T = LC[-1,0]
        T2 = T
    else:
        raise ValueError('Please fetch period for eps='+str(eps)+
                         'and enter it into thalamic_diffs')
    
    #theta_vals = np.linspace(.1,T/2,10)
    
    theta_vals = np.arange(T/20,T,T/20)
    
    for i,theta in enumerate(theta_vals):
        
        fname = (a.dir+'diff_phi='+str(theta)
                 +'_eps='+str(eps)+'Tf='+str(Tf))+'.txt'
        
        file_does_not_exist = not(os.path.isfile(fname))
        
        if file_does_not_exist or recompute:
            phase1 = 0
            phase2 = theta
            
            # transform to indices and get inits
            phase1_idx = int(phase1*len(LC[:,0])/T)
            phase2_idx = int(phase2*len(LC[:,0])/T)
            
            #print(phase1_idx,phase2_idx,len(LC),phase2-phase1)
            #print(LC_small[phase1_idx,:])
            init = np.append(LC[phase1_idx,1:],LC[phase2_idx,1:])
            init[np.array([0,4])] /= 100
            init[np.array([2,6])] *= 100
            
            sol = solve_ivp(rhs,[0,Tf],init,args=(a,eps),method='LSODA',
                            atol=1e-8,rtol=1e-8)
            
            v1_peak_idxs = find_peaks(sol.y[0,:])[0]
            v2_peak_idxs = find_peaks(sol.y[4,:])[0]
            
            # match total number of peaks
            min_idx = np.amin([len(v1_peak_idxs),len(v2_peak_idxs)])
            v1_peak_idxs = v1_peak_idxs[:min_idx]
            v2_peak_idxs = v2_peak_idxs[:min_idx]
            
            phase1 = sol.t[v1_peak_idxs]
            phase2 = sol.t[v2_peak_idxs]
            
            t1 = sol.t[v1_peak_idxs]
            #phi = (phase2-phase1)
            
            #pos = np.where(np.abs(np.diff(phi)) >= 0.25)[0]
            #t1 = np.insert(t1, pos, np.nan)
            #phi[pos] *= -phi[pos] 
            
            #phi = np.mod(phi,1)
            
            #if (phase2-phase1)[-1] > -6 and (phase2-phase1)[-1] < -2:
            #    # collect phase difs ending in antiphase (period 8.4)
                
                
            #else:
            #    # colelct phase diffs ending in snychorony (period 10)
            #    phi = np.mod((phase2-phase1)/T,1)
            #    #phi = (phase2-phase1)/T
            
            data = np.zeros((len(t1),3))
            data[:,0] = t1
            data[:,1] = phase1
            data[:,2] = phase2
            
            np.savetxt(fname,data)
            
        else:
            data = np.loadtxt(fname)
            
        t = data[:,0]
        phi = data[:,2] - data[:,1]
        
        if eps == 0.25 or eps == 0.09:
            if phi[-1] > 0:
                phi *= -1
            
            
        phi_mod = np.mod(phi,T)
        pos = np.where(np.abs(np.diff(phi_mod)) >= 0.25)[0]
        
        t = np.delete(t,pos)
        phi_mod = np.delete(phi_mod,pos)
        
        phi2 = phi_mod/T
        
            
        #ax.plot(data[:,2]-data[:,1],data[:,0],color='k',lw=1)
        ax.plot(phi2,t,color='k',lw=1)
        #ax.plot(t,data[:,2]-np.linspace(0,data[-1,0],len(data[:,0])),color='k',lw=1)
        
        #print(fname)
        #ax.plot(data[:,0],data[:,1],color='k',lw=1)
        ax.set_ylim(data[:,0][-1],data[:,0][0])
            
        ax.set_xticks([0,1/2,1])
        ax.set_xticklabels(['$0$','$T/2$','$T$'])
    return ax

>>>>>>> Stashed changes


def thalamic_h():
    fig = plt.figure()
    
    return fig

def thalamic_1par():
    fig = plt.figure()
    
    return fig

def generate_figure(function, args, filenames, dpi=100):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;

    fig = function(*args)

    if type(filenames) == list:
        for name in filenames:
            if name.split('.')[-1] == 'ps':
                fig.savefig(name, orientation='landscape',dpi=dpi)
            else:
                fig.savefig(name,dpi=dpi,bbox_inches='tight')
    else:
        if name.split('.')[-1] == 'ps':
            fig.savefig(filenames,orientation='landscape',dpi=dpi)
        else:
            fig.savefig(filenames,dpi=dpi)

def main():
    
    # listed in order of Figures in paper
    figures = [
<<<<<<< Updated upstream
        #(cgl_h,[],['cgl_h.pdf','cgl_h.png'])
        #(cgl_2par,[],['cgl_2par.pdf','cgl_2par.png']),
        (thalamic_h,[],['thalamic_h.pdf']),
        #(thalamic_1par,[],['thalamic_1par.pdf']),
=======
        (cgl_h,[],['cgl_h.pdf','cgl_h.png']),
        (cgl_2par,[],['cgl_2par.pdf','cgl_2par.png']),
        
        #(thalamic_h,[],['thal_h.pdf','thal_h.png']),
        #(thalamic_1par,[],['thal_1par.pdf']),
>>>>>>> Stashed changes
    ]
    
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    
    __spec__ = None
    
    main()
