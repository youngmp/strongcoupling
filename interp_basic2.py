# -*- coding: utf-8 -*-
"""
create function-like object for interpolation.

use np.interp for speed as opposed to interp1d. Too much overhead in interp1d.
"""

import numpy as np


class interp_basic(object):
    
    def __init__(self,x,y,T=None):
        
        if T is not None:
            self.T = T
        else:
            self.T = None
        self.x = x
        self.y = y

    def __call__(self, x_new):
        if self.T is not None:
            x_new = np.mod(x_new,self.T)
        return np.interp(x_new, self.x, self.y)
        #return self._call_linear_np(x_new)
