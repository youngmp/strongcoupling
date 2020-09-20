# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:25:22 2020

crappy bifurcation diagram
"""

from Thalamic import Thalamic

import time
import numpy as np

from matplotlib import cm
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
               'trunc_order':6,
               'ib_val':3.5,
               'NA':6000,
               'NB':6000,
               'TN':20000,
               'p_iter':15,
               'load_all':True}
    
    a = Thalamic(**options)
    
    ve_vals = np.linspace(0,1.5,100)
    
    
    
    stability_sync = np.zeros(len(ve_vals))
    stability_anti = np.zeros(len(ve_vals))
    
    #a.load_p()
    a.load_h()
    
    fig = plt.figure(figsize=(4,4))
    
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[:,:])
    #ax2 = fig.add_subplot(gs[0,0])
    #ax3 = fig.add_subplot(gs[1,0])
    
    col = cm.viridis(np.linspace(0,0.5,len(ve_vals)))
    #col = [cm.rainbow(i) for i in np.linspace(0, 1, len(ve_vals))]

    pointx = []
    pointy = []
    stability = []
    
    terms = 7

    for i,ve in enumerate(ve_vals):
        print('i,ve =',i,ve)
        
        #print(a.h_odd_data)
    
        h = 0
        for j in range(terms):
            hterm = a.hodd['dat'][j]
            h += ve**(j+1)*hterm
            
        #ax1.plot(h,color=col[i])
        #plt.show(block=True)
        
        #plt.pause(0.05)
        
        # get zeros
        crossing_idxs = np.where(np.diff(np.sign(h)))[0]
        
        crossing_idxs = np.append(crossing_idxs,0)
        # check stability at each zero
        for crossing_idx in crossing_idxs:
            if crossing_idx == 0:   
                diff = h[1] - h[0]
                #print(diff,h[:10])
            else:
                diff = h[crossing_idx] - h[crossing_idx-1]
                
            
            pointx.append(ve)
            pointy.append(crossing_idx)
            
            if diff > 0:
                stability.append('r')
            else:
                stability.append('k')
            
            
    
    #ax1.set_ylim(-1,1)
    #ax1.scatter(ve_vals,stability_sync)
    #ax1.scatter(ve_vals,stability_anti)
    
    #ax1.plot([0,2],[0,0],color='black',lw=2)
    
    x = np.linspace(0,10.6,len(h))
    
    skipn = 1
    px = pointx[::skipn]
    py = pointy[::skipn]
    
    for i in range(len(px)):
        ax1.scatter(px[i],x[py[i]],color=stability[::skipn][i],s=10)
    
    #ax1.set_ylim(0,10.6)
    
    ax1.set_ylabel('Phase (0 to T. Red=unstable, black=stable)')
    ax1.set_xlabel(r'Coupling strength (eps or gsyn)')
    
    ax1.set_title('terms='+str(terms))
    
    #ax1.set_ylabel(r'Coupling Strength $\varepsilon$')
    #ax1.set_xlabel(r'Coupling parameter $d$')
    
    #ax1.set_title(r'C',x=0)
    #ax2.set_title(r'A',x=0)
    #ax3.set_title(r'B',x=0)
    
    
    #handles, labels = ax1.get_legend_handles_labels()
    #order = [1,2,0]
    #ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    #ax1.legend()
    
    plt.tight_layout()
    
    #fig.colorbar(twopar, ax=ax)
    plt.savefig('thal_bifurcation_zoomed_'+str(terms)+'.png')
    plt.show(block=True)
    
    
if __name__ == '__main__':
    
    __spec__ = None
    main()