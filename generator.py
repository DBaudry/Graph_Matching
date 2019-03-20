import numpy as np
from itertools import combinations
from copy import copy


def generate_random_graph(n, m, draw_attribute=np.random.uniform):
    """
    :param n: number of vertices
    :param m: number of edges
    :param draw_attribute: Distribution for the parameter
    :return: tuple (n, set of edges, attribute matrix A)
    """
    edge_set = list(combinations(np.arange(n), 2))
    edges_indices = np.random.choice(len(edge_set), replace=False, size=m)
    edges = [edge_set[i] for i in edges_indices]
    A = np.zeros((n, n))
    for e in edges:
        A[e[0], e[1]] = draw_attribute()
    A = A+A.T
    A += np.diag(draw_attribute(size=n))
    return n, edges, A


def get_perturbed_graph(graph, noise_level, draw_noise=np.random.uniform):
    n, E, A = graph
    indices = np.arange(n)
    np.random.shuffle(indices)
    Ep = []
    Ap = np.zeros(A.shape)
    for e in E:
        new_e = (indices[e[0]], indices[e[1]])
        Ep.append(new_e)
        Ap[new_e[0], new_e[1]] = A[e[0], e[1]] + draw_noise(0, noise_level)
    Ap = Ap + Ap.T
    for i, x in enumerate(indices):
        Ap[x, x] = A[i, i] + draw_noise(0, noise_level)
    return n, Ep, Ap, indices


def exp_dist(x, y):
    return np.exp(-np.sum(np.abs(x-y)**2))


def get_compatibility_matrix(G1, G2, func):
    n1, edge_1, A1 = G1[:3]
    n2, edge_2, A2 = G2[:3]
    W = np.zeros((n1 * n2, n1 * n2))
    for e1 in edge_1:
        for e2 in edge_2:
            score = func(A1[e1[0], e1[1]], A2[e2[0], e2[1]])
            # each of the four representations of the edges is assigned a fourth of the score
            W[n2*e1[0]+e2[0], n2*e1[1]+e2[1]] = score/4
            W[n2*e1[1]+e2[0], n2*e1[0]+e2[1]] = score/4
            W[n2*e1[0]+e2[1], n2*e1[1]+e2[0]] = score/4
            W[n2*e1[1]+e2[1], n2*e1[0]+e2[0]] = score/4
    for i in range(n1):
        for j in range(n2):
            W[n2*i+j, n2*i+j] = func(A1[i, i], A2[j, j])
    return W


def get_1t1_constraints(n):
    """
    :param n: shape of the affinity matrix
    :return:
    """
    n_tot = n**2
    C0 = np.zeros((n, n_tot))
    C1 = np.zeros((n, n_tot))
    b = np.ones(2*n)
    for i in range(n):
        for k in range(n):
            C0[i, n*i+k] = 1
            C1[i, n*k+i] = 1
    return C0, C1, b


def get_Pc_SMAC(C, b):
    k = C.shape[0]
    Ik = np.zeros((k-1, k))
    Ik[:, :-1] = np.eye(k-1)
    for i in range(k):
        C[i] = C[i] - b[i]/b[-1]*C[-1]
    Ceq = np.dot(Ik, C)
    inv_C = np.linalg.inv(np.dot(Ceq, Ceq.T))
    all_C = np.dot(Ceq.T, np.dot(inv_C, Ceq))
    return np.eye(C.shape[1])-all_C


def random_permutation_matrix(n):
    """
    :param n: shape of the squared matrix
    :return: nxn permutation matrix
    """
    r = np.arange(n)
    np.random.shuffle(r)
    m = np.zeros((n, n))
    for i, x in enumerate(r):
        m[i, x] = 1
    return m.flatten()