import numpy as np
from copy import copy
from skimage.util import view_as_blocks


def bistochastic_normalization(W, n1, n2, max_iter=100, tol=1e-4):
    blocks = view_as_blocks(W, (n1, n1))
    S = np.zeros((n1**2, n2**2))
    for j in range(n2**2):
        S[:, j] = blocks[j//n2, j % n2].flatten('F')
    err = np.inf
    iter = 0
    while iter < max_iter and err > tol:
        old_S = copy(S)
        S = np.apply_along_axis(norm_check, 1, S)
        S = np.apply_along_axis(norm_check, 0, S)
        err = np.sum(np.abs(S-old_S)**2)
        iter += 1
    W_blocks = [[S[:, n2*i+j].reshape((n1, n1)) for i in range(n2)] for j in range(n2)]
    # print(np.dot(S, np.ones((n2*n2))))
    # print(np.dot(S.T, np.ones((n1*n1))))
    return np.block(W_blocks).T


def norm_check(x):
    s = np.abs(x).sum()
    if s > 0:
        return x/s
    else:
        return x
