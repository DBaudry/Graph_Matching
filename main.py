import numpy as np
import generator

if __name__ == '__main__':
    graph1 = generator.generate_random_graph(10, 10)
    graph2 = generator.get_perturbed_graph(graph1, 0.1)
    C = generator.get_compatibility_matrix(graph1, graph2, func=generator.exp_dist)
    print(C)
