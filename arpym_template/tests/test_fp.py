import unittest
import numpy as np
# from numpy.testing import assert_almost_equal, assert_equal, assert_array_equal

from arpym_template.estimation.flexible_probabilities import FlexibleProbabilities



class TestFP(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_mean_cov(self):
        data = np.random.randn(100,2)
        fp = FlexibleProbabilities(data)
        
        err_mean = np.linalg.norm(fp.mean() - np.mean(data, axis=0))
        err_cov = np.linalg.norm(fp.cov() - np.cov(data.T)*0.99)
        
        self.assertAlmostEqual(err_mean, 0)
        self.assertAlmostEqual(err_cov, 0)
 
if __name__ == '__main__':
    unittest.main()
    


