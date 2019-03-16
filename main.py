import numpy as np
import generator
from normalization import bistochastic_normalization
from skimage.util import view_as_blocks


# np.random.seed(123444)
if __name__ == '__main__':
    graph1 = generator.generate_random_graph(5, 10)
    graph2 = generator.get_perturbed_graph(graph1, 0.2)
    W = generator.get_compatibility_matrix(graph1, graph2, func=generator.exp_dist)
    BSN_W = bistochastic_normalization(W, 5, 5)
    print(graph1)
    print(graph2)
    print(W)
    print(BSN_W)


