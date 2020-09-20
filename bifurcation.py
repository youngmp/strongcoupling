# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:25:22 2020

crappy bifurcation diagram

check if diagram looks better or same with and without endpoints in 
pA calculation
"""

from CGL import CGL


import numpy as np

import matplotlib.pyplot as plt


def main():
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
               'TN':2001,
               'load_all':True}
    
    d_center = 1
    a = CGL(**options)
    
    options['load_all'] = False
    
    d_vals = np.linspace(d_center-1,d_center+1,200)[::1]
    dd = (d_vals[1] - d_vals[0])
    #d_vals_original = np.linspace(d_center-.5,d_center+.5,50)[:3]
    #dx = d_vals_original[1]-d_vals_original[0]

    #d_vals_lo = np.arange(0,0.5,dx)
    #d_vals_hi = np.arange(1.5,2,dx)
    
    #d_vals_lo = np.arange(0.5,1,1)
    #d_vals_hi = np.arange(1.5,2,1)
    
    #d_vals = np.append(d_vals_lo,d_vals_original)
    #d_vals = np.append(d_vals,d_vals_hi)
    
    ve_vals = np.linspace(-.8,.4,len(d_vals))
    dve = (ve_vals[1]-ve_vals[0])
    
    fig = plt.figure(figsize=(8,4))
    
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[:,1:])
    ax2 = fig.add_subplot(gs[0,0])
    ax3 = fig.add_subplot(gs[1,0])
    
    stability_sync = np.zeros((len(d_vals),len(ve_vals)))
    stability_anti = np.zeros((len(d_vals),len(ve_vals)))
    
    
    for j,d in enumerate(d_vals):
        print('j,d =',j,d)
        options['d_val'] = d
        
        a.__init__(**options)
        
        a.load_p_sym()
        a.load_p()
        a.load_h_sym()
        a.load_h()
        #print(a.h_odd_data)
        
        for k,ve in enumerate(ve_vals):
            
            
            h = 0
            for i in range(10):
                hterm = a.hodd['dat'][i]
                
                if i == 2:
                    hterm /= 1
                h += ve**(i+1)*hterm
            
            # save specific h functions
            """
            if (d == d_vals[20] and ve == ve_vals[10]):
                
                ax2.plot([0,2*np.pi],[0,0],color='gray')
                ax2.plot(np.linspace(0,2*np.pi,len(h)),h)
                
                #ax2.set_xticks([])
                ax2.set_xticks([0,np.pi,2*np.pi])
                ax2.set_xticklabels([r'$0$',r'$\pi$',r'$2\pi$'])
                                    
                ax2.set_xlabel('Phase angle')
                ax2.set_ylabel('8th Order RHS')
                ax2.set_xlim(0,2*np.pi)
                
                ax1.scatter(d,ve,marker='^',color='tab:blue',s=150)
                ax1.text(d+dd,ve,'A',fontsize=12)
                
            if (d == d_vals[70] and ve == ve_vals[70]):
                ax3.plot([0,2*np.pi],[0,0],color='gray')
                ax3.plot(np.linspace(0,2*np.pi,len(h)),h)
                
                #ax3.set_xticks([])
                ax3.set_xticks([0,np.pi,2*np.pi])
                ax3.set_xticklabels([r'$0$',r'$\pi$',r'$2\pi$'])
                                    
                ax3.set_xlabel('Phase angle')
                ax3.set_ylabel('8th Order RHS')
                ax3.set_xlim(0,2*np.pi)
                
                ax1.scatter(d,ve,marker='*',color='tab:blue',s=150)
                ax1.text(d+dd,ve,'B',fontsize=12)

                #plt.tight_layout()
                #plt.savefig('h_d='+str(d)+'.pdf')
                #plt.close()
            """
            #ax.set
            
            # get zeros
            diff0 = h[1] - h[0]
            diff1 = h[int(len(h)/2)] - h[int(len(h)/2)-1]
            
            if diff0 > 0:
                stability_sync[j,k] = 1
            else:
                stability_sync[j,k] = -1
                
            if diff1 > 0:
                stability_anti[j,k] = 2
            else:
                stability_anti[j,k] = -2
                
                
    
    # process data for plotting 2 parameter  diagram.        
    
    
    
    # transpose and flip as below puts matrix in agreement with axis coordinates.
    #twopar = ax.imshow(stability_sync.T[::-1,:]+stability_anti.T[::-1,:],
    #                   aspect='auto',
    #                   extent=[d_vals[0],d_vals[-1],
    #                           ve_vals[0],ve_vals[-1]])
    
    total_mat = stability_sync.T[::-1,:]+stability_anti.T[::-1,:]
    
    diff1 = np.abs(np.diff(total_mat,axis=0))
    diff2 = np.abs(np.diff(total_mat,axis=1))
    
    ddiff1 = diff1[:,1:]*diff2[1:,:]
    ddiff2 = diff1[:,1:]*diff2[:-1,:]
    ddiff3 = 0#diff1[:,:-1]*diff2[1:,:]
    ddiff4 = 0#diff1[:,:-1]*diff2[:-1,:]
    
    diff_final = ddiff1 + ddiff2 + ddiff3 + ddiff4
    
    #twopar = ax.imshow(diff_final,aspect='auto',
    #                   extent=[d_vals[0],d_vals[-1],
    #                           ve_vals[0],ve_vals[-1]])
    
    
    # get matrix index pairs where diff is nonzero
    r_idxs, c_idxs = np.where(np.diff(diff_final) > 0)
    nonzero_idxs = zip(r_idxs,c_idxs)
    
    #print(nonzero_idxs)
    eps_plots = np.zeros(len(r_idxs))
    d_plots = np.zeros(len(c_idxs))
    
    for i, (j,k) in enumerate(nonzero_idxs):
        eps_plots[i] = ve_vals[np.mod(len(ve_vals)-j,len(ve_vals))]-dve
        d_plots[i] = d_vals[k]+2*dd
    
    
    # order and create lines
    q1_condition = (d_plots > 1)*(eps_plots > 0)
    q2_condition = (d_plots < 1)*(eps_plots > 0)
    q3_condition = (d_plots < 1)*(eps_plots < 0)
    q4_condition = (d_plots > 1)*(eps_plots < 0)
    
    
    sorted_idx1 = np.argsort(d_plots[q1_condition])
    sorted_idx2 = np.argsort(d_plots[q2_condition])
    sorted_idx3 = np.argsort(d_plots[q3_condition])
    sorted_idx4 = np.argsort(d_plots[q4_condition])
    
    color_8th = 'tab:red'
    lw = 4
    alpha = 0.7
    
    ax1.plot([0,2],[0,0],color='black',lw=2)
    
    
    # I add by dd and dve instead of modifying the index.
    # this isn't cheating -- the error is accounted for by 
    # the relatively coarse discretization. so errors off by a factor
    # the spatial interval are to be expected.
    
    ax1.plot(d_plots[q1_condition][sorted_idx1],
            eps_plots[q1_condition][sorted_idx1],
            color=color_8th,label='9th Order',lw=lw,alpha=alpha)
    
    ax1.plot(d_plots[q2_condition][sorted_idx2],
            eps_plots[q2_condition][sorted_idx2],
            color=color_8th,lw=lw,alpha=alpha,ls='--')
    
    ax1.plot(d_plots[q3_condition][sorted_idx3],
            eps_plots[q3_condition][sorted_idx3]-dve,
            color=color_8th,lw=lw,alpha=alpha)
    
    ax1.plot(d_plots[q4_condition][sorted_idx4],
            eps_plots[q4_condition][sorted_idx4],
            color=color_8th,lw=lw,alpha=alpha,ls='--')
    
    
    #ax.plot(d_plots[d_plots>1][sorted_idx],eps_plots[d_plots>1][sorted_idx])
    
    #ax.plot(d_plots[d_plots>1][sorted_idx],eps_plots[d_plots>1][sorted_idx])
    
    #ax.scatter(d_plots[::-1],eps_plots[::-1],s=5,color='red',label='8th Order')
    
    # true cruve stability
    es_true = (d_vals*a.q_val-1)/(d_vals**2+1)
    ea_true = (1-d_vals*a.q_val)/(d_vals**2-2*a.q_val*d_vals + 3)
    
    es_appx = (d_vals*a.q_val-1)/(d_vals**2*(1+a.q_val**2))
    ea_appx = (1-d_vals*a.q_val)/(d_vals**2*(1+a.q_val**2))
    
    color_PRL = 'blue'
    color_true = 'black'
    
    ax1.plot(d_vals,es_true,color=color_true,label='True',lw=2)
    ax1.plot(d_vals,ea_true,color=color_true,ls='--',lw=2)
    
    ax1.plot(d_vals,es_appx,color=color_PRL,label='2nd Order')
    ax1.plot(d_vals,ea_appx,color=color_PRL,ls='--')
    
    # label regions
    ax1.text(1.1,0.25,'I',fontsize=16,family='serif')
    ax1.text(1.1,-0.3,'I',fontsize=16,family='serif')

    ax1.text(0.25,0.02,'II',fontsize=16,family='serif')
    ax1.text(1.6,-0.08,'II',fontsize=16,family='serif')
    
    ax1.text(1.6,0.02,'III',fontsize=16,family='serif')
    ax1.text(0.25,-0.08,'III',fontsize=16,family='serif')
    
    ax1.set_ylim(ve_vals[0],ve_vals[-1])
    ax1.set_xlim(d_vals[0],d_vals[-1])
    
    ax1.set_ylabel(r'Coupling Strength $\varepsilon$')
    ax1.set_xlabel(r'Coupling parameter $d$')
    
    ax1.set_title(r'C',x=0)
    ax2.set_title(r'A',x=0)
    ax3.set_title(r'B',x=0)
    
    
    handles, labels = ax1.get_legend_handles_labels()
    order = [1,2,0]
    ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    #ax1.legend()
    
    plt.tight_layout()
    
    #fig.colorbar(twopar, ax=ax)
    plt.savefig('cgl_twopar.pdf')
    plt.show(block=True)
    
    
if __name__ == '__main__':
    
    __spec__ = None
    main()
