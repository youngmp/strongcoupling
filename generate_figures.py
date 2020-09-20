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

from matplotlib.legend_handler import HandlerBase

from CGL import CGL

from scipy.optimize import brentq

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

def cgl_2par(recompute_2par=False,recompute_all=False):
    
    fig = plt.figure(figsize=(5,4))
        
    gs = fig.add_gridspec(1, 1)    
    ax = fig.add_subplot(gs[:,:])
    
    # use this array and cut out pieces to avoid lots of recomputing.
    d_center = 1
    d_vals = np.linspace(d_center-1,d_center+1,100)[10:][::2]
    
    options = {'recompute_g_sym':False,
               'recompute_g':False,
               'recompute_het_sym':False,
               'recompute_z':False,
               'recompute_i':False,
               'recompute_k_sym':False,
               'recompute_p_sym':False,
               'recompute_p':recompute_all,
               'recompute_h_sym':False,
               'recompute_h':recompute_all,
               'trunc_order':9,
               'trunc_derviative':2,
               'd_val':1,
               'q_val':1,
               'TN':2001,
               'load_all':True}
    
    q_val = options['q_val']
    #d_val = options['d_val']
    
    fname = ('twopar_cgl'
             +'_order='+str(options['trunc_order'])
             +'.txt')
    
    file_not_found = not(os.path.isfile(fname))
    
    if recompute_2par or file_not_found:
        a = CGL(**options)
        options['load_all'] = False
        
        # first column d, second col sync, third column anti.
        ve_sync_anti = np.zeros((len(d_vals),3))
        ve_sync_anti[:,0] = d_vals
        
        for j,d in enumerate(d_vals):
            print('j,d =',j,d)
            options['d_val'] = d
            
            a.__init__(**options)
            
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
        #(cgl_h,[],['cgl_h.pdf','cgl_h.png'])
        #(cgl_2par,[],['cgl_2par.pdf','cgl_2par.png']),
        (thalamic_h,[],['thalamic_h.pdf']),
        #(thalamic_1par,[],['thalamic_1par.pdf']),
    ]
    
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    
    __spec__ = None
    
    main()
