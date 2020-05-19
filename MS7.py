import numpy as np
from scipy.stats import laplace
from tabulate import tabulate
import scipy.stats as stats
import math as m

alpha = 0.05
p = 1 - alpha
k = 6

size = 40
# distribution = np.random.normal(0, 1, size=100)
distribution = laplace.rvs(size=size, scale=1 / m.sqrt(2), loc=0)

mu = np.mean(distribution)
sigma = np.std(distribution)

print('mu = ' + str(np.around(mu, decimals=2)))
print('sigma = ' + str(np.around(sigma, decimals=2)))

limits = np.linspace(-1.1, 1.1, num=k - 1)
chi_2 = stats.chi2.ppf(p, k-1)

print('chi_2 = ' + str(chi_2))

p_list = np.array([])
n_list = np.array([])
for i in range(-1, len(limits)):
    prev_cdf_val = stats.norm.cdf(limits[i]) if i != -1 else 0
    cur_cdf_val = stats.norm.cdf(limits[i+1]) if i != len(limits) - 1 else 1
    p_list = np.append(p_list, cur_cdf_val - prev_cdf_val)
    if i == -1:
        n_list = np.append(n_list, len(distribution[distribution <= limits[0]]))
    elif i == len(limits) - 1:
        n_list = np.append(n_list, len(distribution[distribution >= limits[-1]]))
    else:
        n_list = np.append(n_list, len(distribution[(distribution <= limits[i + 1]) & (distribution >= limits[i])]))

result = np.divide(np.multiply((n_list - size * p_list), (n_list - size * p_list)), p_list * size)

cols = ["i", "limits", "n_i", "p_i", "np_i", "n_i - np_i", "/frac{(n_i-np_i)^2}{np_i}"]
rows = []
for i in range(0, len(n_list)):
    if i == 0:
        boarders = ['-inf', np.around(limits[0], decimals=2)]
    elif i == len(n_list) - 1:
        boarders = [np.around(limits[-1], decimals=2), 'inf']
    else:
        boarders = [np.around(limits[i - 1], decimals=2), np.around(limits[i], decimals=2)]

    rows.append([i + 1, boarders, n_list[i], np.around(p_list[i], decimals=4), np.around(p_list[i] * size, decimals=2),
                 np.around(n_list[i] - size * p_list[i], decimals=2), np.around(result[i], decimals=2)])

rows.append([len(n_list), "-", np.sum(n_list), np.around(np.sum(p_list), decimals=4),
             np.around(np.sum(p_list * size), decimals=2),
             -np.around(np.sum(n_list - size * p_list), decimals=2),
             np.around(np.sum(result), decimals=2)])
print(tabulate(rows, cols, tablefmt="latex"))

print(len(n_list))
print('\n')
