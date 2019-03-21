import numpy as np
import expe as xp
import matplotlib.pyplot as plt

np.random.seed(12)
if __name__ == '__main__':
    n, m, sigma = 20, 40, 1
    # n, m, sigma = 3, 2, 0.
    print(xp.random_xp(10, n, m, noise_lvl=sigma, dists=('g', 'g')))
    # noise_map = np.arange(0, 6, 0.5)
    # res = xp.xp_noise_map(noise_map, 100, n, m)
    # X = np.linspace(0, 6, len(noise_map))
    # plt.plot(X, res[0]['SMAC'], label='SMAC')
    # plt.plot(X, res[0]['SMACb'], label='SMAC with norm')
    # plt.legend()
    # plt.show()
