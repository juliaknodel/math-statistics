from collections import defaultdict

from scipy.stats import norminvgauss, laplace, poisson, cauchy, uniform
import numpy as np
import matplotlib.pyplot as plt
import math as m

sizes = [20, 100]
count = 1000
names = ['n', 'l', 'p', 'c', 'u']


def moustache(distr):
    q_1, q_3 = np.quantile(distr, [0.25, 0.75])
    return q_1 - 3 / 2 * (q_3 - q_1), q_3 + 3 / 2 * (q_3 - q_1)


def distributions(size):
    n = norminvgauss.rvs(1, 0, size=size)
    l = laplace.rvs(size=size, scale=1 / m.sqrt(2), loc=0)
    p = poisson.rvs(10, size=size)
    c = cauchy.rvs(size=size)
    u = uniform.rvs(size=size, loc=-m.sqrt(3), scale=2 * m.sqrt(3))
    counted_distributions = [n, l, p, c, u]
    return counted_distributions


def count_out(distr):
    x1, x2 = moustache(distr)
    filtered = [x for x in distr if x > x2 or x < x1]
    return len(filtered)


for size in sizes:
    d = defaultdict(float)
    distribution = distributions(size)
    # для рисования
    # for distr in distribution:
    #     plt.boxplot(distr)
    #     plt.show()
    for i in range(1000):
        distribution = distributions(size)
        for k in range(len(distribution)):
            distr = distribution[k]
            name = names[k]
            d[name] += count_out(distr)
    for key in d:
        d[key] = d[key]/size*(1/1000)
    print(d)
