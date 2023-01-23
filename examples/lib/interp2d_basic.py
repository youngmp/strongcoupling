import numpy as np
import scipy.interpolate as si

class interp2d_basic(object):
    """
    create interp2d object that allows vector inputs
    also inputs are mod T.
    
    Requires interp2d object already be made.
    """
    
    def __init__(self,fn,T):
        self.T = T
        self.fn = fn
        self.tck = fn.tck

    def __call__(self, x_new, y_new):
        x_new = np.mod(x_new,self.T)
        y_new = np.mod(y_new,self.T)
        
        
        #tck0 = self.fn.tck[0]
        #tck1 = self.fn.tck[1]
        #tck2 = self.fn.tck[2]
        #tck3 = self.fn.tck[3]
        #tck4 = self.fn.tck[4]
        
        #return si.dfitpack.bispeu(self.tck[0],
        #                          self.tck[1],
        #                          self.tck[2],
        #                          self.tck[3],
        #                          self.tck[4],x_new,y_new)[0]
    
        return self.bispeu(self.fn,x_new,y_new)
    
    def bispeu(self,fn,x,y):
        """
        silly workaround
        https://stackoverflow.com/questions/47087109/...
        evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
        """
        return si.dfitpack.bispeu(fn.tck[0], fn.tck[1],
                                  fn.tck[2], fn.tck[3],
                                  fn.tck[4], x, y)[0]