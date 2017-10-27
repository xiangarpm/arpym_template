#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:43:33 2017

@author: xiangshi
"""

import numpy as np
from cvxopt import matrix, solvers, spmatrix


def min_rel_entropy(p_pri, v_ineq, v_eq, m_ineq, m_eq):
    """min_p sum p*log(p/p_pri)
        s.t. a_ineq*p <= b_ineq
    """

    v = np.concatenate((v_ineq, v_eq), axis=0)
    m = np.concatenate((m_ineq, m_eq), axis=0)
    k_ineq = len(m_ineq)
    k_eq = len(m_eq)
    k_ = k_ineq + k_eq

    G = spmatrix(1.0, range(k_ineq), range(k_ineq), (k_ineq, k_))
    h = matrix(0.0, (k_ineq, 1))

    def F(x=None, z=None):
        if x is None:
            return 0, matrix(1.0, (k_, 1))

        theta = np.array(x.T)[0, :]
        p = p_pri * np.exp(np.dot(theta, v))
        phi = np.log(np.sum(p))  # log partition
        p = p/np.sum(p)  # posterior
        f = phi - np.dot(theta, m)  # Lagrangian
        dphi = np.matrix(np.dot(v, p))  # gradient of the log partition
        grad = matrix(dphi - m)

        if z is None:
            return f, grad

        v_ = v - dphi.T
        H = matrix(np.dot(np.multiply(v_, p), v_.T))
        return f, grad, H

    sol = solvers.cp(F, G, h)
    theta = np.array(sol['x'].T)[0, :]
    p = p_pri * np.exp(np.dot(theta, v))
    p = p/np.sum(p)

n = 100
p_pri = np.ones(n) / n
v_ineq = np.random.randn(5, n)
m_ineq = np.zeros(5)
v_eq = np.random.randn(5, n)
m_eq = np.zeros(5)
