from scipy.stats import norminvgauss, laplace, poisson, cauchy, uniform
import numpy as np
import matplotlib.pyplot as plt
import math as m

sizes = [10, 50, 1000]

rv_n = norminvgauss(1, 0)
rv_l = laplace(scale=1 / m.sqrt(2), loc=0)
rv_p = poisson(10)
rv_c = cauchy()
rv_u = uniform(loc=-m.sqrt(3), scale=2 * m.sqrt(3))

densities = [rv_n, rv_l, rv_p, rv_c, rv_u]
names = ["Normal", "Laplace", "Poisson", "Cauchy", "Uniform"]

for size in sizes:
    n = norminvgauss.rvs(1, 0, size=size)
    l = laplace.rvs(size=size, scale=1 / m.sqrt(2), loc=0)
    p = poisson.rvs(10, size=size)
    c = cauchy.rvs(size=size)
    u = uniform.rvs(size=size, loc=-m.sqrt(3), scale=2 * m.sqrt(3))
    distributions = [n, l, p, c, u]
    build = list(zip(distributions, densities, names))
    for histogram, density, name in build:
        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram, density=True, histtype='stepfilled', alpha=0.6, color="green")
        if histogram is p:
            x = np.arange(poisson.ppf(0.01, 10), poisson.ppf(0.99, 10))
            ax.plot(x, density.pmf(x), 'k-', lw=2)
        else:
            x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
            ax.plot(x, density.pdf(x), 'k-', lw=2)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title("Size: " + str(size))
        plt.show()




