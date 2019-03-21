import numpy as np
from tqdm import tqdm
import generator as gn
from normalization import bistochastic_normalization
import solver as sl
from discretize import discretize_one_to_one as dct1


def random_xp(n_expe, n, m, noise_lvl, dists=('u', 'u')):
    dist_attribute = np.random.uniform if dists[0] == 'u' else np.random.normal
    dist_noise = np.random.uniform if dists[1] == 'u' else np.random.normal
    param_att = (-1, 1) if dists[0] == 'u' else (0, 1)
    param_noise = (-noise_lvl/2, noise_lvl/2) if dists[1] == 'u' else (0, noise_lvl)
    C, b = gn.get_1t1_constraints(n)
    Pc = gn.get_Pc_SMAC(C, b)
    score_perfect_match = {'SM': 0, 'GA': 0, 'SMAC': 0, 'SMb': 0, 'GAb': 0, 'SMACb': 0}
    score = {'SM': 0, 'GA': 0, 'SMAC': 0, 'SMb': 0, 'GAb': 0, 'SMACb': 0}
    avg_regret = {'SM': 0, 'GA': 0, 'SMAC': 0, 'SMb': 0, 'GAb': 0, 'SMACb': 0}

    for _ in tqdm(range(n_expe)):

        # Generate new graphs and get compatibility matrix
        graph1 = gn.generate_random_graph(n, m, param_att, dist_attribute)
        graph2 = gn.get_perturbed_graph(graph1, param_noise, dist_noise)
        W = gn.get_compatibility_matrix(graph1, graph2, func=gn.exp_dist)
        BSN_W = bistochastic_normalization(W, n, n)
        BSN_W = n*BSN_W

        # Compute xp
        true_P = get_true_permutation_matrix(graph2[-1], n)
        SM = sl.SM(W)
        GA = sl.GA_matrix(W, n, x0=gn.random_permutation_matrix(n))
        SMAC = sl.SMAC(W, Pc)
        SM, GA, SMAC = dct1(W, n, SM), dct1(W, n, GA, is_GA=True), dct1(W, n, SMAC)

        SM_b = sl.SM(BSN_W)
        GA_b = sl.GA_matrix(BSN_W, n, x0=gn.random_permutation_matrix(n))
        SMAC_b = sl.SMAC(BSN_W, Pc)
        SM_b, GA_b, SMAC_b = dct1(BSN_W, n, SM_b), dct1(BSN_W, n, GA_b, is_GA=True), dct1(BSN_W, n, SMAC_b)

        names = ['SM', 'GA', 'SMAC', 'SMb', 'GAb', 'SMACb']
        res = [SM, GA, SMAC, SM_b, GA_b, SMAC_b]
        # update score
        for i, name in enumerate(names):
            score_perfect_match[name] += (np.abs((res[i]-true_P)).sum() == 0)/n_expe
            score[name] += (res[i]*true_P).sum()/n/n_expe
            true_obj = np.dot(true_P, np.dot(W, true_P))
            avg_regret[name] += (true_obj - np.dot(res[i], np.dot(W, res[i])))/n_expe
    for x in score.keys():
        score[x] = round(score[x], 3)
        score_perfect_match[x] = round(score_perfect_match[x], 3)
    return score, score_perfect_match, avg_regret


def get_true_permutation_matrix(permut_list, n):
    P = np.zeros((n, n))
    for i, x in enumerate(permut_list):
        P[i, x] = 1
    return P.flatten()


def xp_noise_map(noise_map, n_expe, n, m, dist_attribute=np.random.uniform, dist_noise=np.random.uniform):
    results = {'SM': [], 'GA': [], 'SMAC': [], 'SMb': [], 'GAb': [], 'SMACb': []}
    results_perfect_match = {'SM': [], 'GA': [], 'SMAC': [], 'SMb': [], 'GAb': [], 'SMACb': []}
    for noise in tqdm(noise_map):
        res = random_xp(n_expe, n, m, noise, dist_attribute, dist_noise)
        for x in results.keys():
            results[x].append(1-res[0][x])
            results_perfect_match[x].append(1-res[1][x])
    return results, results_perfect_match
