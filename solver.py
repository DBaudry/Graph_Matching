import numpy as np
from scipy.linalg import eigh
from copy import copy
import cvxopt as cvx
from ubsdp import ubsdp
import picos as pic


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


def GA_matrix(W, n, x0, b0=0.5, bf=10., br=1.075,
              tol0=0.5, tol1=0.05, itermax0=30, itermax1=30, display_step=False):
    """
    :param W: Affinity matrix
    :param n: size the graphs
    :param x0: starting assignment
    :param b0: starting control parameter
    :param bf: upper value of the control parameter
    :param br: increase rate for the control parameter
    :param tol0: tolerance for convergence of the gradient descent on bistochastic x
    :param tol1: tolerance for bistochastic normalization of x
    :param itermax0: maximal number of iterations for the gradient descent on bistochastic x
    :param itermax1: maximal number of iterations for a single normalization
    :param display_step: show x for each control parameter
    :return: bistochastic approximation of the solution of 1 to 1 matching
    """
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


def SMAC(W, Pc):
    """
    :param W: Affinity Matrix
    :param Pc: Matrix obtained by transforming the constraint matrix, according to the paper
    :return: leading eigenvector of Pc*W*Pc
    """
    new_W = np.dot(Pc, np.dot(W, Pc))
    return SM(new_W)


def SDP_relax(W, n):
    """
    Ne marche pas pour l'instant
    :param W:
    :param n:
    :return:
    """
    d = np.diag(W)
    Weq = np.zeros((W.shape[0]+1, W.shape[1]+1))
    Weq[0, 1:] = d/2
    Weq[1:, 0] = d/2
    Weq[1:, 1:] = W-np.diag(d)

    prob = pic.Problem()
    X = prob.add_variable('X', (n**2+1, n**2+1), vtype="symmetric")
    W_new = pic.new_param('W_new', cvx.matrix(Weq))
    prob.set_objective('max', (X | W_new))

    c, A = [], []

    # A1 = pic.new_param('A1', cvx.matrix(A_11))
    # prob.add_constraint((A1 | X) == 1)
    # print(prob)

    # First condition: X[0, 0] = 1
    A_11 = np.zeros(Weq.shape)
    A_11[0, 0] = 1
    c.append(1), A.append(A_11)

    # Second condition: weak enforcement of x_i = x_i^2
    for i in range(W.shape[0]):
        Ai = np.zeros(Weq.shape)
        Ai[0, i] = -1
        Ai[i, 0] = -1
        Ai[i, i] = 2
        c.append(0), A.append(Ai)

    # Conditions of Sum_{j}(x_i*n+j) for all i, same for columns
    for s in range(n):
        As = np.zeros(Weq.shape)
        Asp = np.zeros(Weq.shape)
        As[n*s+1:n*(s+1)+1, n*s+1:n*(s+1)+1] = np.ones((n, n))
        for t in range(n):
            if s > 0:
                Asp[(n*t) % s + 1, (n*t) % s + 1] = 1
            else:
                Asp[n*t+1, n*t+1] = 1
        c.append(1), A.append(As)
        c.append(1), A.append(Asp)

    n_const = copy(len(c))
    c = tuple(c)
    pic.new_param('c', c)
    pic.new_param('A', A)
    prob.add_list_of_constraints([(A[i] | X) == c[i] for i in range(n_const)])
    print(prob)
    prob.solve(verbose=0, solver='cvxopt')
    print(X)
    return X