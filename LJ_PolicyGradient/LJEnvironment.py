import numpy as np
from IPython import embed

from fingerprintFeature import *
from doubleLJ import *

class LJEnvironment():
    """
    """
    
    def __init__(self, *params):
        """
        """
        self.params = params[0]
        self.fpf = fingerprintFeature(rcut = 4,binwidth=0.2)
        
        
    def gridToXY(self,gridCoord):
        """
        """

        r0 = self.params[0]

        a1 = gridCoord[0]
        a2 = gridCoord[1]    

        v1 = np.array([r0,0])
        v2 = np.array([np.cos(np.pi/3)*r0,np.sin(np.pi/3)*r0])
        xy = a1*v1 + a2*v2

        return np.array(xy)        

    def getEnergy(self,XY):
        """
        """
        
        N = XY.shape[0]

        xlist = XY.T[0]
        ylist = XY.T[1]        
        
        X = np.zeros(2*N)

        for i in range(N):
            X[2*i] = xlist[i]
            X[2*i+1] = ylist[i]    

        return doubleLJ_energy(X,*self.params)

    def getFeature(self,XY):
        """
        """


        N = XY.shape[0]

        xlist = XY.T[0]
        ylist = XY.T[1]        
        
        X = np.zeros(2*N)

        for i in range(N):
            X[2*i] = xlist[i]
            X[2*i+1] = ylist[i]    

        
        return self.fpf.get_singleFeature(X)
