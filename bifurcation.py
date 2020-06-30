# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:25:22 2020

crappy bifurcation diagram
"""

from CGL import CGL


import numpy as np

import matplotlib.pyplot as plt


def main():
    # for NIC, 3rd derivatives go away, so we only need trunc_gh=3.
    a = CGL(recompute_g_sym=False,
            recompute_g=False,
            recompute_het_sym=False,
            recompute_z=False,
            recompute_i=False,
            recompute_k_sym=False,
            recompute_p_sym=False,
            recompute_p=False,
            recompute_h_sym=False,
            recompute_h=False,
            trunc_order=2,
            trunc_derviative=2,
            d_val=1,
            q_val=2*np.pi)
    
    ve_range = np.linspace(.05,.2,50)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    stable = []
    unstable = []
    
    for ve in ve_range:
        h = 0
        for i in range(2):
            h += ve**(i+1)*a.h_odd_data[i]
            
                
        # get zeros
        idxs = np.arange(len(a.B_array))
        crossing_bools = np.where(np.diff(np.signbit(h)))[0]  # (np.diff(np.sign(h)) != 0)
        crossing_idxs = idxs[crossing_bools]
        
        if 0 not in crossing_idxs:
            crossing_idxs = np.append(crossing_idxs[::-1],0)[::-1]
            
            
        # get stability
        for i,index in enumerate(crossing_idxs):
            #print('i,index,ve',i,index,ve)
            try:
                if index > 0:
                    diff = h[index] - h[index-1]
                else:
                    diff = h[index+1] - h[index]
                    
                    
                if diff > 0:
                    unstable.append([ve,index])
                else:
                    stable.append([ve,index])
                    
                    
            except IndexError:
                print('ignored index',index)
            
        #ax.scatter(idxs,h)
        #ax.plot([0,len(h)],[0,0],color='gray')
        #ax.scatter(idxs[crossing_idxs],np.zeros(len(idxs[crossing_idxs])),
        #           color='red',s=10,zorder=10)
        #print(stable)
    
    #print(stable)
    #print(unstable)
    
    stable = np.asarray(stable)
    unstable = np.asarray(unstable)
    
    stable[:,1] = a.B_array[stable[:,1].astype(int)]
    unstable[:,1] = a.B_array[unstable[:,1].astype(int)]
    
    ax.scatter(stable[:,0],stable[:,1],color='black',s=10)
    ax.scatter(unstable[:,0],unstable[:,1],color='red',s=10)
    
    plt.show(block=True)
    
    
if __name__ == '__main__':
    main()