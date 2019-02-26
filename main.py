import numpy as np
import generator
from normalization import bistochastic_normalization


np.random.seed(12344)
if __name__ == '__main__':
    graph1 = generator.generate_random_graph(20, 40)
    graph2 = generator.get_perturbed_graph(graph1, 0.2)
    W = generator.get_compatibility_matrix(graph1, graph2, func=generator.exp_dist)
    BSN_W = bistochastic_normalization(W, 20, 20)



