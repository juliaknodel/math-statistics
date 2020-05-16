import numpy as np
import scipy.stats as stats
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import statistics

sizes = [20, 60, 100]
rhos = [0, 0.5, 0.9]


def show_ellipse(size):
    fig, ax = plt.subplots(1, 3)
    size_str = "n = " + str(size)
    titles = [size_str + r', $ \rho = 0$', size_str + r', $\rho = 0.5 $', size_str + r', $ \rho = 0.9$']
    for i in range(len(rhos)):
        num, rho = i, rhos[i]
        rv = get_multi_normal(size, rho)
        x, y = rv[:, 0], rv[:, 1]
        get_ellipse(x, y, ax[num], edgecolor='pink')
        ax[num].scatter(x, y, s=5)
        ax[num].set_title(titles[num])
    plt.savefig("n" + str(size) + ".png", format='png')
    plt.show()


def get_tex_table(pearson_coefs, spearman_coefs, quadrant_coefs, rho, size):
    if rho != -1:
        rows = [["rho = " + str(rho), 'r', 'r_{S}', 'r_{Q}']]
    else:
        rows = [["size = " + str(size), 'r', 'r_{S}', 'r_{Q}']]
    p = np.median(pearson_coefs)
    s = np.median(spearman_coefs)
    q = np.median(quadrant_coefs)
    rows.append(['E(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = np.median([pearson_coefs[k] ** 2 for k in range(1000)])
    s = np.median([spearman_coefs[k] ** 2 for k in range(1000)])
    q = np.median([quadrant_coefs[k] ** 2 for k in range(1000)])
    rows.append(['E(z^2)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    p = statistics.variance(pearson_coefs)
    s = statistics.variance(spearman_coefs)
    q = statistics.variance(quadrant_coefs)
    rows.append(['D(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])

    return tabulate(rows, [], tablefmt="latex")


def get_quadrant(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    n = {1: 0, 2: 0, 3: 0, 4: 0}
    for i in range(size):
        if x[i] >= med_x and y[i] >= med_y:
            n[1] += 1
        elif x[i] < med_x and y[i] >= med_y:
            n[2] += 1
        elif x[i] < med_x and y[i] < med_y:
            n[3] += 1
        elif x[i] >= med_x and y[i] < med_y:
            n[4] += 1
    return (n[1] + n[3] - n[2] - n[4])/size


def get_coefficients(size, rho, get_rvs):
    pearson_coef = []
    spearman_coef = []
    quadrant_coef = []
    for i in range(1000):
        rv = get_rvs(size, rho)
        x, y = rv[:, 0], rv[:, 1]
        pearson_coef.append(stats.pearsonr(x, y)[0])
        spearman_coef.append(stats.spearmanr(x, y)[0])
        quadrant_coef.append(get_quadrant(x, y))
    return pearson_coef, spearman_coef, quadrant_coef


def get_ellipse(x, y, ax, n_std=3.0, **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_multi_normal(size, rho):
    return stats.multivariate_normal.rvs([0, 0], [[1.0, rho], [rho, 1.0]], size=size)


def get_mix_multi_normal(size, rho):
    return 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + 0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)


for size in sizes:
    for rho in rhos:
        pearson_coef, spearman_coef, quadrant_coef = get_coefficients(size, rho, get_multi_normal)
        print('\n' + str(size) + '\n' + str(get_tex_table(pearson_coef, spearman_coef, quadrant_coef, rho, size)))

    pearson_coef, spearman_coef, quadrant_coef = get_coefficients(size, 0, get_mix_multi_normal)
    print('\n' + str(size) + '\n' + str(get_tex_table(pearson_coef, spearman_coef, quadrant_coef, -1, size)))
    show_ellipse(size)
