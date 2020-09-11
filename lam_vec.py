# -*- coding: utf-8 -*-
"""
create function-like object for heterogeneous functions.

use np.interp for speed as opposed to interp1d. Too much overhead in interp1d.
"""

import numpy as np


class lam_vec(object):
    
    def __init__(self,lam_list):
        self.lam_list = lam_list

    def __call__(self, t):
        return np.array(list(map(lambda f: f(t),self.lam_list)))
                