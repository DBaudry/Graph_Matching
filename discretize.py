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
    print('Greedy procedure failed to provide a solution')
    return x.flatten() #debug code below later

    X = (x.flatten() > 0.5).reshape((n, n)).astype('float64')  #Links we are sure about
    row_ind = np.where(X.sum(axis=1) == 0)
    col_ind = np.where(X.sum(axis=0) == 1)
    current_obj = -np.inf
    for k in range(n_sampling):
        new_X = copy(X)
        for i in range(len(row_ind)):
            test_row = list(copy(row_ind))
            test_col = list(copy(col_ind))
            new_r = np.random.choice(test_row)
            new_c = np.random.choice(test_col, p=X[new_r, test_col]/X[new_r, test_col].sum())
            new_X[new_r, new_c] = 1
            test_row.remove(new_r)
            test_col.remove(new_c)

        new_x = new_X.flatten()
        new_obj = np.dot(x.T, np.dot(W, x))
        if new_obj > current_obj:
            x = new_x
            current_obj = new_obj
    return x
