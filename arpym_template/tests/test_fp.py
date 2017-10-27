# -*- coding: utf-8 -*-
import unittest
import numpy as np

from arpym_template.estimation.flexible_probabilities import FlexibleProbabilities
from arpym_template.toolbox.min_rel_entropy import min_rel_entropy


class TestFP(unittest.TestCase):

    def setUp(self):
        pass

    def test_mean_cov(self):
        data = np.random.randn(100, 2)
        fp = FlexibleProbabilities(data)

        err_mean = np.linalg.norm(fp.mean() - np.mean(data, axis=0))
        err_cov = np.linalg.norm(fp.cov() - np.cov(data.T)*0.99)

        self.assertAlmostEqual(err_mean, 0)
        self.assertAlmostEqual(err_cov, 0)

    def test_mre(self):

        fp_pri = FlexibleProbabilities(np.ones(4))

        a_eq = np.array([[1.,  1.,  1.,  1.], [0.,  1.,  1.,  1.]])
        b_eq = np.array([[1], [0.6]])

        fp_pos = min_rel_entropy(fp_pri, None, None, a_eq, b_eq)[0]

        err = np.linalg.norm(fp_pos.p -
                             np.array([[0.4000006, 0.2000002,
                                        0.2000002, 0.2000002]]))

        self.assertAlmostEqual(err, 0)


if __name__ == '__main__':
    unittest.main()
