# -*- coding: utf-8 -*-
"""
For details, see
`Section 3.1 <https://www.arpm.co/lab/redirect.php?permalink=setting-flexible-probabilities>`_.
"""
from collections import namedtuple
import numpy as np


class FlexibleProbabilities(object):
    """Flexible Probabilities
    """
    def __init__(self, data):
        self.x = data
        self.p = np.ones(len(data))/len(data)

    def shape(self):
        """Shape of the data
        """
        return self.x.shape

    def mean(self):
        """Sample mean with flexible probabilities
        """
        return np.dot(self.p, self.x)

    def cov(self):
        """Sample covariance with flexible probabilities
        """
        x_ = self.x - np.mean(self.x, axis=0)
        return np.dot(np.multiply(np.transpose(x_), self.p), x_)

    def equal_weight(self):
        """Equally weighted probabilities
        """
        self.p = np.ones(len(self.x))/len(self.x)

    def exponential_decay(self, tau):
        """Exponentail decay probabilities
        """
        t_ = len(self.x)
        self.p = np.exp(-np.log(2)/tau*(t_-np.arange(0, t_)))
        self.p = self.p / np.sum(self.p)

    def smooth_kernel(self, z=None, z_star=None, h=None, gamma=2):
        """Smooth kernel probabilities
        """
        if z is None:
            z = self.x[:, 0]

        if z_star is None:
            z_star = np.mean(z)

        if h is None:
            h = np.std(z)

        self.p = np.exp(-(np.abs(z - z_star)/h)**gamma)
        self.p = self.p / np.sum(self.p)

    def effective_scenarios(self, Type=None):
        """This def computes the Effective Number of Scenarios of Flexible
        Probabilities via different types of defs

        For details on the function, please see
        |ex_effective_scenarios| |code_effective_scenarios|

        Note:
            The exponential of the entropy is set as default, otherwise specify
            ``Type.ExpEntropy.on = true`` to use the exponential of the entropy
            or specify ``Type.GenExpEntropy.on = true`` and supply the scalar
            ``Type.ExpEntropy.g`` to use the generalized exponential of the
            entropy.

        Args:
            Type (tuple): type of def: ``ExpEntropy``, ``GenExpEntropy``

        Returns:
            ens (double): Effective Number of Scenarios

        .. |ex_effective_scenarios| image:: icon_ex_inline.png
            :scale: 20 %
            :target: https://www.arpm.co/lab/redirect.php?permalink=EBEffectNbScenFun

        .. |code_effective_scenarios| image:: icon-code-1.png
            :scale: 20 %
            :target: https://www.arpm.co/lab/redirect.php?code=EffectiveScenarios
        """
        if Type is None:
            Type = namedtuple('type', ['Entropy'])
            Type.Entropy = 'Exp'
        if Type.Entropy != 'Exp':
            Type.Entropy = 'GenExp'

        # Code
        p_ = self.p
        if Type.Entropy == 'Exp':
            p_[p_ == 0] = 10 ** (-250)  # avoid log(0) in ens computation
            ens = np.exp(-p_@np.log(p_.T))
        else:
            ens = np.sum(p_ ** Type.g) ** (-1 / (Type.g - 1))

        return ens


def diff_length_mlfp(fp, nu, threshold, smartinverse=0, maxiter=10**5):
    """Maximum-likelihood with flexible probabilities for different-length
    series

    For details on the function, please see
    |ex_diff_length_mlfp| |code_diff_length_mlfp|

    Note:
        We suppose the missing values, if any, are at the beginning.
        (the farthest observations in the past could be missing).
        We reshuffle the series in a nested pattern, such that the series with
        the longer history comes first and the one with the shorter history
        comes last.

    Args:
        fp (FlexibleProbabilities): obsrevations with flexible probabilities
        nu (double): degrees of freedom for the multivariate Student
            t-distribution
        threshold (double): convergence thresholds
        smartinverse (double, optional): additional parameter: set it to 1 to
            use LRD smart inverse in the regression process
        maxiter (int, optional): maximum number of iterations inside
            ``MaxLikFPTReg``

    Returns:
        mu (numpy.ndarray): DLFP estimate of the location parameter
        sig2 (numpy.ndarray): DLFP estimate of the dispersion parameter

    .. |ex_diff_length_mlfp| image:: icon_ex_inline.png
        :scale: 20 %
        :target: https://www.arpm.co/lab/redirect.php?permalink=DiffLengthRout

    .. |code_diff_length_mlfp| image:: icon-code-1.png
        :scale: 20 %
        :target: https://www.arpm.co/lab/redirect.php?codeplay=DiffLengthMLFP
    """
    return None
