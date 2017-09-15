from collections import namedtuple
import pandas as pd
import numpy as np
from scipy.stats import norm

class FlexibleProbabilities(object):
    """Flexible Probabilities

    Attributes:
        x: 
        p: 
    """
    def __init__(self, data):
        self.x = data
        self.p = np.ones(len(data))/len(data)

        
    def shape(self):
        return self.x.shape
    
    
    def mean(self):
        """
        Sample mean with flexible probabilities
        """
        return np.dot(self.p, self.x)
    
    
    def cov(self):
        """
        Sample covariance with flexible probabilities
        """
        x_ = self.x - np.mean(self.x, axis=0)
        return np.dot(np.multiply(np.transpose(x_), self.p), x_)
        
        
    def equal_weight(self):
        """
        Equally weighted probabilities
        """
        self.p = np.ones(len(self.x))/len(self.x)
        
        
    def exponential_decay(self, tau):
        """
        Exponentail decay probabilities
        """
        t_ = len(self.x)
        self.p = np.exp(-np.log(2)/tau*(t_-np.arange(0,t_)))
        self.p = self.p /np.sum(self.p)
        
        
    def smooth_kernel(self, z=None, z_star=None, h=None, gamma=2):
        """
        Smooth kernel probabilities
        """
        if z is None:
            z = self.x[:,0]
            
        if z_star is None:
            z_star = np.mean(z)
            
        if h is None:
            h = np.std(z)
            
        self.p = np.exp(-(np.abs(z - z_star)/h)**gamma)
        self.p = self.p /np.sum(self.p)

        
    def effective_scenarios(self, Type=None):
        """
        This def computes the Effective Number of Scenarios of Flexible
        Probabilities via different types of defs
        INPUTS
        p       : [vector] (1 x j_) vector of Flexible Probabilities
        Type    : [struct] type of def: 'ExpEntropy', 'GenExpEntropy'
        OUTPUTS
        ens     : [scalar] Effective Number of Scenarios
        NOTE:
        The exponential of the entropy is set as default, otherwise
        Specify Type.ExpEntropy.on = true to use the exponential of the entropy
        or
        Specify Type.GenExpEntropy.on = true and supply the scalar
        Type.ExpEntropy.g to use the generalized exponential of the entropy

        For details on the exercise, see here: https://www.arpm.co/lab/redirect.php?permalink=EBEffectNbScenFun
        """
        if Type is None:
            Type = namedtuple('type',['Entropy'])
            Type.Entropy = 'Exp'
        if Type.Entropy != 'Exp':
            Type.Entropy = 'GenExp'

        ## Code
        p_ = self.p
        if Type.Entropy == 'Exp':
            p_[p_==0] = 10**(-250)    #avoid log(0) in ens computation
            ens = np.exp(-p_@np.log(p_.T))
        else:
            ens = np.sum(p_ ** Type.g) ** (-1 / (Type.g - 1))

        return ens
        
    
