import numpy as np
from solver import GA_matrix
from copy import copy


def discretize_one_to_one(W, n, x_res, is_GA=False, n_sampling = 100):
    """
    :param W: Affinity matrix
    :param x: Result from the first algorithm
    :param is_GA: True if the first algorithm is Graduated Assignment
    :return: Discretized solution for one to one matching (i.e a Permutation Matrix)
    """
    if not is_GA:
        x = GA_matrix(W, n, x0=x_res)
    else:
        x = copy(x_res)

    # Check if the solution is already a permutation matrix
    row_sum = np.dot(x.reshape((n, n)), np.ones(n))
    col_sum = np.dot(x.reshape((n, n)).T, np.ones(n))
    diff_row, diff_col = row_sum - np.ones(n), col_sum - np.ones(n)
    if np.abs(diff_row).sum() == 0 and np.abs(diff_col).sum() == 0:
        print('Output of GA is already permutation matrix')
        return x

    # Greedy procedure: if the max on rows is the max on columns then this permutation is in the solution
    x = x.reshape((n, n))
    max_row = np.argmax(x, axis=1)
    max_col = np.argmax(x, axis=0)
    row_solution = np.zeros((n, n))
    col_solution = np.zeros((n, n))
    for i in range(n):
        row_solution[i, max_row[i]] = 1
        col_solution[max_col[i], i] = 1
    diff = np.abs(row_solution-col_solution).sum()
    if diff == 0:
        print('Greedy procedure successful: no conflict between rows and columns')
        return row_solution.flatten()

    # Randomized method based on Sampling

    X = row_solution*col_solution
    unfilled_row = np.where(X.sum(axis=1) == 0)[0]
    unfilled_col = np.where(X.sum(axis=0) == 0)[0]
    if unfilled_row.shape[0] == 1:
        X[unfilled_row[0], unfilled_row[0]] = 1
        return X.flatten()

    prob = x[unfilled_row][:, unfilled_col]
    prob = np.apply_along_axis(lambda x: x/x.sum(), 1, prob)
    r = np.arange(len(unfilled_row))
    c = np.arange(len(unfilled_col))

    current_obj = -np.inf
    for k in range(n_sampling):
        new_X = copy(X)
        proba_row = np.ones(len(unfilled_row))/len(unfilled_row)
        p = copy(prob)
        res_r = []
        res_c = []
        for i in range(len(r)):
            new_r = np.random.choice(r, p=proba_row)
            new_c = np.random.choice(c, p=p[new_r])
            res_r.append(new_r)
            res_c.append(new_c)
            proba_row[new_r] = 0
            proba_row = proba_row/proba_row.sum()
            p[:, new_c] = 0
            p = np.apply_along_axis(lambda x: x/x.sum(), 1, p)

        for i in range(len(unfilled_row)):
            new_X[unfilled_row[res_r[i]], unfilled_col[res_c[i]]] = 1
        new_x = new_X.flatten()
        new_obj = np.dot(new_x.T, np.dot(W, new_x))
        if new_obj > current_obj:
            x = new_x
            current_obj = new_obj
    return x
