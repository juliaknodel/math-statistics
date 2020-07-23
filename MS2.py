from scipy.stats import norminvgauss, laplace, poisson, cauchy, uniform
import numpy as np
import math as m


def z_np(distr, np):
    if np.is_integer():
        return distr[int(np)]
    else:
        return distr[int(np)+1]


def z_q(distr, size):
    z_1 = z_np(distr, size/4)
    z_2 = z_np(distr, 3*size/4)
    return (z_1+z_2)/2


def z_r(distr, size):
    return (distr[0]+distr[-1])/2


def z_tr(distr, size):
    r = int(size/4)
    sum = 0
    for i in range(r+1, size-r+1):
        sum += distr[i]
    return (1/(size-2*r))*sum


def mean(distr, size):
    return np.mean(distr)


def median(distr, size):
    return np.median(distr)


sizes = [10, 100, 1000]
count = 1000
names = ["Normal", "Laplace", "Poisson", "Cauchy", "Uniform"]

characteristics = [[[], [], [], [], []],
                   [[], [], [], [], []],
                   [[], [], [], [], []]]
mean_characteristics = [[[], [], [], [], []],
                        [[], [], [], [], []],
                        [[], [], [], [], []]]
mean_characteristics_sq = [[[], [], [], [], []],
                           [[], [], [], [], []],
                           [[], [], [], [], []]]

functions = [mean, median, z_r, z_q, z_tr]

for i in range(len(sizes)):
    size = sizes[i]
    for k in range(count+1):
        n = norminvgauss.rvs(1, 0, size=size)
        l = laplace.rvs(size=size, scale=1 / m.sqrt(2), loc=0)
        p = poisson.rvs(10, size=size)
        c = cauchy.rvs(size=size)
        u = uniform.rvs(size=size, loc=-m.sqrt(3), scale=2 * m.sqrt(3))
        distributions = [n, l, p, c, u]
        for num in range(len(distributions)):
            d = distributions[num]
            for s in range(len(functions)):
                if k == 0:
                    characteristics[i][num].append([])
                f = functions[s]
                value = f(d, size)
                characteristics[i][num][s].append(value)
    for num in range(len(distributions)):
        for s in range(len(functions)):
            distr = characteristics[i][num][s]
            mean_characteristics[i][num].append(np.mean(distr))
            mean_characteristics_sq[i][num].append(np.std(distr)*np.std(distr))
for i in range(len(sizes)):
    print('\n')
    print(sizes[i])
    print('\n')
    for num in range(len(distributions)):
        if num != 2:
            continue
        print('\n')
        print(names[num])
        print('\n')
        for s in range(len(functions)):
            print(str(mean_characteristics[i][num][s]) + " " + str(mean_characteristics_sq[i][num][s]))
            print(' ')


