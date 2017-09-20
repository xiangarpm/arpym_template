#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:15:21 2017

@author: xiangshi
"""
import numpy as np

import arpym_template

# Equal weighted FP
fp = arpym_template.estimation.FlexibleProbabilities(np.random.randn(5,2))
print(fp.x, fp.p, fp.mean(), fp.cov(), fp.effective_scenarios())

# Exponential decay FP
fp.exponential_decay(2)
print(fp.x, fp.p, fp.mean(), fp.cov(), fp.effective_scenarios())

# Minimum relative entropy test
a_eq = np.array([[ 1.,  1.,  1.,  1., 1.], [ 0.,  1.,  1.,  1., 0.]])
b_eq = np.array([[1], [0.6]])
fp_pos = arpym_template.toolbox.min_rel_entropy(fp, None, None, a_eq, b_eq)[0]