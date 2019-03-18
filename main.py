import numpy as np
import generator
from normalization import bistochastic_normalization
from skimage.util import view_as_blocks
import solver as sl
from tqdm import tqdm

np.random.seed(12)
if __name__ == '__main__':
    n, m, sigma = 3, 2, 0.2
    for k in tqdm(range(10)):
        graph1 = generator.generate_random_graph(n, m)
        graph2 = generator.get_perturbed_graph(graph1, sigma)
        W = generator.get_compatibility_matrix(graph1, graph2, func=generator.exp_dist)
        BSN_W = bistochastic_normalization(W, n, n)
        # print(graph1)
        # print(graph2)
        print('True permutation: {}'.format(graph2[-1]))
        print(sl.SM(W).reshape((n, n)))
        print(sl.GA_matrix(W, n, x0=sl.random_permutation_matrix(n)).reshape((n, n)))



