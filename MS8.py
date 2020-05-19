import numpy as np
import scipy.stats as stats

gamma_conf = 0.95

sizes = [20, 100]


def get_m_conf_interval_normal(distr, size):
    mean, sigma = np.mean(distr), np.std(distr)
    interval = sigma * stats.t.ppf((1 + gamma_conf) / 2, size - 1) / (size - 1) ** 0.5

    return np.around(mean - interval, decimals=2), np.around(mean + interval, decimals=2)


def get_sigma_conf_interval_normal(distr, size):
    sigma = np.std(distr)
    low_lim = sigma * (size / stats.chi2.ppf((1 + gamma_conf) / 2, size - 1)) ** 0.5
    up_lim = sigma * (size / stats.chi2.ppf((1 - gamma_conf) / 2, size - 1)) ** 0.5

    return np.around(low_lim, decimals=2), np.around(up_lim, decimals=2)


def get_m_conf_interval_random(distr, size):
    mean, sigma = np.mean(distr), np.std(distr)
    u = stats.norm.ppf((1 + gamma_conf) / 2)
    interval = sigma * u / (size ** 0.5)

    return np.around(mean - interval, decimals=2), np.around(mean + interval, decimals=2)


def get_sigma_conf_interval_random(distr, size):
    mean, sigma = np.mean(distr), np.std(distr)

    m_4 = stats.moment(distr, 4)
    e = m_4 / (sigma**4) - 3
    u = stats.norm.ppf((1 + gamma_conf) / 2)
    U = u * (((e + 2) / size) ** 0.5)

    low_lim = sigma * (1 + 0.5 * U) ** (-0.5)
    up_lim = sigma * (1 - 0.5 * U) ** (-0.5)

    return np.around(low_lim, decimals=2), np.around(up_lim, decimals=2)


for size in sizes:
    distribution = np.random.normal(0, 1, size=size)
    print('size = ' + str(size))

    a, b = get_m_conf_interval_normal(distribution, size)
    len = b-a
    print('normal m: (' + str(a) + ', ' + str(b) + ') ' + 'len: ' + str(len))

    a, b = get_m_conf_interval_random(distribution, size)
    len = b - a
    print('random m: (' + str(a) + ', ' + str(b) + ') ' + 'len: ' + str(len))

    a, b = get_sigma_conf_interval_normal(distribution, size)
    len = b - a
    print('normal sigma: (' + str(a) + ', ' + str(b) + ') ' + 'len: ' + str(len))

    a, b = get_sigma_conf_interval_random(distribution, size)
    len = b - a
    print('random sigma: (' + str(a) + ', ' + str(b) + ') ' + 'len: ' + str(len))
