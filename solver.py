import numpy as np
from scipy.linalg import eigh
from copy import copy
from normalization import bistochastic_normalization


def SM(W):
    """
    :param W: affinity matrix
    :return: leading eigenvector of W if W is non-negative
    """
    l, v = eigh(W)
    v = np.round(v, 6)
    s = v.sum(axis=0)
    for i in range(v.shape[1]):  #remove spurious results
        if s[i] == 0:
            l[i] = 0
    lead_ind = np.argmax(l)
    lead_ev = v[:, lead_ind]
    return np.abs(lead_ev)


def random_permutation_matrix(n):
    r = np.arange(n)
    np.random.shuffle(r)
    m = np.zeros((n, n))
    for i, x in enumerate(r):
        m[i, x] = 1
    return m.flatten()


def GA_matrix(W, n, x0, b0=0.5, bf=10., br=1.075,
              tol0=0.5, tol1=0.05, itermax0=30, itermax1=30, display_step=False):
    beta, x = b0, x0
    while beta < bf:
        err, n_iter = np.inf, 0
        while err > tol0 and n_iter < itermax0:
            prev_x = copy(x)
            Q = np.dot(W, x)
            x = np.exp(beta*Q)
            err1, n_iter1 = np.inf, 0
            x = x.reshape((n, n))
            while n_iter1 < itermax1 and err1 > tol1:
                prev_x1 = copy(x)
                x = np.apply_along_axis(lambda y: y/y.sum(), 0, x)
                x = np.apply_along_axis(lambda y: y/y.sum(), 1, x)
                err1 = np.sum(np.abs(x-prev_x1)**2)
                n_iter1 += 1
            x = x.flatten()
            x = np.round(x, 6)
            err = np.sum(np.abs(x-prev_x)**2)
            n_iter += 1
        beta *= br
        if display_step:
            print('beta {}'.format(beta))
            print(x.reshape((n, n)))
    return np.round(x, 3)


def GA(W, x0, b0, bf, br, graph1, graph2, tol0, tol1, itermax0, itermax1):
    n1, E1, A1 = graph1
    n2, E2, A2 = graph2
    beta = b0
    attrib_1 = [A1[e[0], e[1]] for e in E1]
    attrib_2 = [A2[e[0], e[1]] for e in E2]
    while beta < bf:
        beta *= br
    pass