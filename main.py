import numpy as np
import expe as xp

np.random.seed(12)
if __name__ == '__main__':
    n, m, sigma = 20, 40, 1
    # n, m, sigma = 3, 2, 0.
    print(xp.random_xp(10, n, m, noise_lvl=sigma))

