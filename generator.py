import numpy as np
from itertools import combinations
from copy import copy


def generate_random_graph(n, m):
    """
    :param n: number of vertices
    :param m: number of edges
    :return: tuple (n, set of edges, attribute matrix A)
    """
    edge_set = list(combinations(np.arange(n), 2))
    edges_indices = np.random.choice(len(edge_set), replace=False, size=m)
    edges = [edge_set[i] for i in edges_indices]
    A = np.zeros((n, n))
    for e in edges:
        A[e[0], e[1]] = np.random.uniform()
    A = A+A.T
    A += np.diag(np.random.uniform(size=n))
    return n, edges, A


def get_perturbed_graph(graph, noise_level):
    n, E, A = graph
    indices = np.arange(n)
    np.random.shuffle(indices)
    Ep = []
    Ap = np.zeros(A.shape)
    for e in E:
        new_e = (indices[e[0]], indices[e[1]])
        Ep.append(new_e)
        Ap[new_e[0], new_e[1]] = A[e[0], e[1]] + np.random.uniform(low=0, high=noise_level)
    Ap = Ap + Ap.T
    for i, x in enumerate(indices):
        Ap[x, x] = A[i, i] + np.random.uniform(low=0, high=noise_level)
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



