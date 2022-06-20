

import numpy as np


class dimension:

    def __init__(self,npix2=10,dmin2=0,dmax2=9,name="dim"):

        self._name = name
        self._npix = npix2
        self._dmin = dmin2
        self._dmax = dmax2

    def pnts(self):
        p = (np.arange(self.n)*(self.max - self.min)/self.n) + self.min
        return p


    """
    @property
    def name(self):
        return self._name


    @property
    def npix(self):
        return self._npix

    @property
    def dmin(self):
        return self._dmin

    @property
    def dmax(self):
        return self._dmax
    """        
    
    

class Vol2:

    #def __init__(self,dimnames=[],dimlen=[],dmin=[],dmax=[]):
    def __init__(self,dimnames=None,dimlen=None,dmin=None,dmax=None):

        self.ndims = len(dimnames)
        if (len(dimlen)!=self.ndims) or (len(dimnames)!=self.ndims) \
                or (len(dmin)!=self.ndims) or (len(dmax)!=self.ndims):
            print( "incorrect initialization of Vol - all arguments must be of same length")
            exit()

        self.dims = {}
        for i in np.arange(self.ndims):
            d = dimension( dimlen[i], dmin[i], dmax[i], dimnames[i] )
            self.dims.update( {d._name : d} ) 

        self.vol = np.zeros( dimlen ) 
        
