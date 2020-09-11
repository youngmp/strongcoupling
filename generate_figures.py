"""
Generate figures for strong coupling paper
"""

#from decimal import Decimal
#from matplotlib.collections import PatchCollection

import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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


def cgl_2par():
    pass



def main():
    
    # listed in order of Figures in paper
    figures = [
        
        (cgl_2par,[],["cgl_2par.pdf"]),
        (thalamic_1par,[],["thalamic_1par.pdf"]),
    ]
    
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    main()
