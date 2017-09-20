from numpy import log, exp, ones, zeros, eye, array, maximum
from scipy.optimize import minimize
import numpy as np
from arpym_template.estimation.flexible_probabilities import FlexibleProbabilities

np.seterr(all='ignore')


def min_rel_entropy(fp_pri, v_ineq, mu_ineq, v_eq, mu_eq, options=None):
    '''This def computes the FP-updated distribution according to linear
    constraints on Flexible Probabilities via Entropy Pooling
    
    Args:
        fp_pri (FlexibleProbabilities): prior distribution
        vx_ineq (array): pick matrix for combinations in inequality views
        mu_ineq (array): vector that quantifies inequality views
        vx_eq (array): pick matrix for combinations in equality views
        mu_eq (array): vector that quantifies equality views
        options (:obj:`dict`, optional): optimization options created with "optimoptions"
    
    Returns:
        fp_pos (FlexibleProbabilities): posterior distribution
        lg_ (array) optimal parameters of dual Lagrangian

    For details on the exercise, see here: https://www.arpm.co/lab/redirect.php?permalink=EB_EntropyPoolFunc
    '''
    
    
    if v_ineq is not None:
        k_ = v_ineq.shape[0]
    else:
        k_ = 0
    if v_eq is not None:
        l_ = v_eq.shape[0]
    else:
        l_ = 0
    options = {'ftol':1e-16, 'disp': False, 'maxiter': 10**6}
    ## Code
    lv0 = zeros((k_ + l_, 1))   # initial point

    fp_pos = FlexibleProbabilities(fp_pri.x)
    
    if k_==0:   # equality constraints
        # cons = {'type': 'eq', 'fun': lambda x, alpha, beta: -alpha.dot(x)+beta, 'args': (v_eq,mu_eq)}
        # res = minimize(mDualLagrangian_eq, lv0, args=(p_pri, v_eq, mu_eq, l_),constraints=cons, options=options)
        options = {'disp': False, 'maxiter': 10**6}
        res = minimize(mDualLagrangian_eq, lv0, args=(fp_pri.p, v_eq, mu_eq), options=options)
        if res.status !=0:
            res = minimize(mDualLagrangian_eq, lv0, args=(fp_pri.p, v_eq, mu_eq), method='Nelder-Mead',options=options)
        v_ = res.x
        fp_pos.p = exp(log(fp_pri.p) - 1 - v_.dot(v_eq))
        lg_ = v_
    else: # inequality and equality constraints
        # specify constraints l >= 0
        alpha = -eye(k_ + l_)
        alpha[k_:, :] = 0
        beta = zeros(k_+l_)
        res = minimize(mDualLagrangian, lv0, args=(fp_pri.p, v_eq, mu_eq, v_ineq, mu_ineq, k_),
                   constraints={'type': 'ineq','fun': lambda x, alpha, beta: -alpha.dot(x)+beta,'args':(alpha,beta)},options=options)

        lg_ = res.x
        l_ = lg_[:k_]
        v_ = lg_[k_:]
        fp_pos.p = exp(log(fp_pri.p) - 1 - l_.T@v_ineq - v_.T@v_eq)
        
    return fp_pos, lg_


def mDualLagrangian_eq(v, fp_pri, a, b):
    '''
    Opposite dual Lagrangian for equality constraints
    '''
    if v.ndim == 1:
        v = v.reshape(-1,1)
        vt = v.T
    else:
        vt = v
    # at = a.T

    p = exp(log(fp_pri.p) - 1 - vt@a)
    p = maximum(p, 10**(-32))
    pt = p.T

    # dual Lagrangian
    h = (log(p) - log(fp_pri))@pt + vt@(a@pt - b)

    mh = -h.squeeze() # value

    # mgrad = b - a@pt # gradient
    # mHess = (a*(ones((l_, 1))@p))@at    # Hessian: a * diag(p) * a.T
    return mh


def ineqcons(x, A, b):
    y = -A*x+b
    return y


def mDualLagrangian(lv, p_pri, a, b, c, d, k_):
    '''
    Opposite dual Lagrangian for inequality and equality constraints

    a: v_eq
    b: mu_eq
    c: v_ineq
    d: mu_ineq
    k_: k_
    l_: dummy
    '''
    l = lv[:k_]
    v = lv[k_:]
    lt = l.T
    vt = v.T

    p = exp(log(p_pri) - 1 - lt@c - vt@a)
    p = maximum(p, 10**(-32))
    pt = p.reshape((-1,1))

    # dual Lagrangian
    h = (log(p) - log(p_pri))@pt + lt@(c@pt - d) + vt@(a@pt - b)

    mh = - h.squeeze() # value
    # mgrad = r_[d.reshape((1,1)), b] - r_[c, a]@pt # gradient
    # return mh, mgrad
    return mh


def mDualLagrangiangrad(lv, p_pri, a, b, c, d, k_):
    '''
    Gradient opposite dual Lagrangian for inequality and equality constraints
    '''
    l = lv[:k_]
    v = lv[k_:]
    lt = l.T
    vt = v.T

    p = exp(log(p_pri) - 1 - lt@c - vt@a)
    p = maximum(p, 10**(-32))
    pt = p.T

    mgrad = array([d,b]) - array([c, a])@pt # gradient
    return mgrad



def mHessianFun1(lv, p_pri, a, b, c, d, k_, l_):
    '''
    Hessian of opposite dual Lagrangian for inequality and equality constraints
    '''
    l = lv[:k_]
    v = lv[k_:]
    lt = l.T
    vt = v.T
    ct = c.T
    at = a.T

    p = exp(log(p_pri) - 1 - lt@c - vt@a)
    p = maximum(p, 10**(-32))

    mHess = (array([c, a])*(ones((k_ + l_, 1))@p))@array([ct, at]) # Hessian: [c a] * diag(p) * [c a].T
    return mHess

