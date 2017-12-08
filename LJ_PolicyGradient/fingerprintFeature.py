import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist
from scipy.special import erf
import time

class fingerprintFeature():
    def __init__(self, X=None, dim=2, rcut=4, binwidth=0.01, sigma=0.2, nsigma=4):
        
        """
        --input--
        X:                                                                                                                                     Contains data. Each row tepresents a structure given by cartesian coordinates
        in the form [x1, y1, ... , xN, yN]
    
        Rcut:
        Cutoff radius
        
        deltaR:
        radial binsize
        
        sigma:
        standard deviation for gaussian smearing
        
        nsigma:
        The distance (as the number of standard deviations sigma) at                                                            
        which the gaussian smearing is cut off (i.e. no smearing beyond                                                         
        that distance). 

        """
        self.X = X
        self.dim=dim
        self.rcut = rcut
        self.binwidth = binwidth
        self.sigma = sigma
        self.nsigma = nsigma

        # parameters for the binning:
        self.m = int(np.ceil(self.nsigma*self.sigma/self.binwidth))  # number of neighbour bins included.
        self.smearing_norm = erf(0.25*np.sqrt(2)*self.binwidth*(2*self.m+1)*1./self.sigma)  # Integral of the included part of the gauss
        self.Nbins = int(np.ceil(rcut/binwidth))

        # Cutoff volume
        self.cutoffVolume = 4*np.pi*self.rcut**2

    def get_singleFeature(self, x):
        """
              --input--
                      x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
                              """
        R = self.radiusVector(x)

        # Number of interatomic distances in the structure
        N_distances = R.shape[0]
        # filter distances longer than rcut + nsigma*sigma
        R = R[R <= self.rcut + self.nsigma*self.sigma]
        
        fingerprint = np.zeros(self.Nbins)
        for deltaR in R:
            rbin = int(np.floor(deltaR/self.binwidth))
            binpos = (deltaR % self.binwidth) / self.binwidth  # From 0 to binwidth (set constant at 0.5*binwidth for original)
            rabove = int(binpos > 0.5)

            # Lower and upper range of bins affected by the current atomic distance deltaR.
            minbin_lim = -self.m-(1-rabove)
            maxbin_lim = self.m+1+rabove
            for i in range(minbin_lim, maxbin_lim):
                newbin = rbin + i  # maybe abs() to make negative bins contribute aswell.
                if newbin < 0 or newbin >= self.Nbins:
                    continue
                
                c = 0.25*np.sqrt(2)*self.binwidth*1./self.sigma
                if i == minbin_lim:
                    erfarg_low = -(self.m+0.5)
                    erfarg_up = i+(1-binpos)
                elif i == maxbin_lim-1:
                    erfarg_low = i-binpos
                    erfarg_up = self.m+0.5
                else:
                    erfarg_low = i-binpos
                    erfarg_up = i+(1-binpos)
                value = 0.5*erf(2*c*erfarg_up)-0.5*erf(2*c*erfarg_low)
                        
                # divide by smearing_norm
                value /= self.smearing_norm
                value /= (4*np.pi*deltaR**2)/self.cutoffVolume * self.binwidth * N_distances
                fingerprint[newbin] += value
        return fingerprint

    def get_featureMat(self, X):
        """
                                      Calculated the feature matrix based on a position matrix 'X'.
                                              ---input---
                                                      X:
                                                              Position matrix with each row 'x' containing the atomic coordinates of
                                                                      a structure. x = [x0,y0,x1,y1, ...].
                                                                              """
        featureMat = np.array([self.get_singleFeature(x) for x in X])
        return featureMat

    def get_singleGradient(self, x):
        """
                                                                                      --input--
                                                                                              x: atomic positions for a single structure in the form [x1, y1, ... , xN, yN]
                                                                                                      """
        Natoms = int(x.shape[0]/self.dim)

        R, dxMat, indexMat = self.radiusVector_grad(x)
        # Number of interatomic distances in the structure
        N_distances = R.shape[0]
        # filter distances longer than rcut + nsigma*sigma
        filter = R <= self.rcut + self.nsigma*self.sigma
        R = R[filter]
        dxMat = dxMat[filter]
        indexMat = indexMat[filter]
        
        fingerprint_grad = np.zeros((self.Nbins, self.dim*Natoms))
        for deltaR, dx, index in zip(R, dxMat, indexMat):
            rbin = int(np.floor(deltaR/self.binwidth))
            binpos = (deltaR % self.binwidth) / self.binwidth  # From 0 to binwidth (set constant at 0.5*binwidth for original)
            rabove = int(binpos > 0.5)

            # Lower and upper range of bins affected by the current atomic distance deltaR.
            minbin_lim = -self.m-(1-rabove)
            maxbin_lim = self.m+1+rabove
            for i in range(minbin_lim, maxbin_lim):
                newbin = rbin + i  # maybe abs() to make negative bins contribute aswell.
                if newbin < 0 or newbin >= self.Nbins:
                    continue

                c = 0.25*np.sqrt(2)*self.binwidth*1./self.sigma
                if i == minbin_lim:
                    arg_low = -(self.m+0.5)
                    arg_up = i+(1-binpos)
                elif i == maxbin_lim:
                    arg_low = i-binpos
                    arg_up = self.m+0.5
                else:
                    arg_low = i-binpos
                    arg_up = i+(1-binpos)
                value1 = -1./deltaR*(erf(2*c*arg_up)-erf(2*c*arg_low))
                value2 = -2*(np.exp(-(2*c*arg_up)**2) - np.exp(-(2*c*arg_low)**2))  # 2 in front..
                value = value1 + value2

                # divide by smearing_norm
                value /= self.smearing_norm
                value /= (4*np.pi*deltaR**2)/self.cutoffVolume * self.binwidth * N_distances

                # Add to the the gradient matrix
                fingerprint_grad[newbin, self.dim*index[0]:self.dim*index[0]+self.dim] += value/deltaR*dx
                fingerprint_grad[newbin, self.dim*index[1]:self.dim*index[1]+self.dim] += -value/deltaR*dx
        return fingerprint_grad

    def get_gradientMat(self, X):
        """
                                                                                                              Calculated the feature=gradient matrix based on a position matrix 'X'.
                                                                                                                      ---input---
                                                                                                                              X:
                                                                                                                                      Position matrix with each row 'x' containing the atomic coordinates of
                                                                                                                                              a structure. x = [x0,y0,x1,y1, ...].
                                                                                                                                                      """
        feature_gradMat = np.array([self.get_singleGradient(x) for x in X])
        return feature_gradMat
        
    def radiusMatrix(self, x):
        """
                                                                                                                                                              Calculates the matrix consisting of all pairwise euclidean distances.
                                                                                                                                                                      """
        Natoms = int(x.shape[0])
        x = x.reshape((Natoms, self.dim))
        R = np.array([[euclidean(xi, xj) for xj in x] for xi in x])
        return R
    
    def radiusVector(self, x):
        """
                                                                                                                                                                              Calculates the vector consisting of all pairwise euclidean distances.
                                                                                                                                                                                      """
        Natoms = int(x.shape[0]/self.dim)
        x = x.reshape((Natoms, self.dim))
        return pdist(x, metric='euclidean')
    
    def radiusVector_grad(self, x):
        """
                                                                                                                                                                                              Calculates the vector consisting of all pairwise euclidean distances.
                                                                                                                                                                                                      """
        Natoms = int(x.shape[0]/self.dim)
        Ndistances = int(Natoms*(Natoms-1)/2)
        x = x.reshape((Natoms, self.dim))
        Rvec = np.zeros(Ndistances)
        dxMat = np.zeros((Ndistances, self.dim))
        indexMat = np.zeros((Ndistances, 2)).astype(int)
        k = 0
        for i in range(Natoms):
            for j in range(i+1, Natoms):
                Rvec[k] = euclidean(x[i],x[j])
                dxMat[k,:] = x[i] - x[j]
                #if self.dim == 2:
                #    dxMat[k,:] = np.array([x[i,0] - x[j,0] , x[i,1] - x[j,1]])
                #if self.dim == 3:
                #    dxMat[k,:] = np.array([x[i,0] - x[j,0] , x[i,1] - x[j,1], x[i,2] - x[j,2]])
                indexMat[k,:] = np.array([i,j])
                k += 1
        return Rvec, dxMat, indexMat
