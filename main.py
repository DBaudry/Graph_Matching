import numpy as np
import expe as xp

np.random.seed(12)
if __name__ == '__main__':
    n, m, sigma = 20, 40, 0.6
    print(xp.random_xp(20, n, m, noise_lvl=sigma))

