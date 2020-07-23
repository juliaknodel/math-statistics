from collections import defaultdict

from scipy.stats import norminvgauss, laplace, poisson, cauchy, uniform
import numpy as np
import matplotlib.pyplot as plt
import math as m


def get_functions():
    rv_n = norminvgauss(1, 0)
    rv_l = laplace(scale=1 / m.sqrt(2), loc=0)
    rv_p = poisson(10)
    rv_c = cauchy()
    rv_u = uniform(loc=-m.sqrt(3), scale=2 * m.sqrt(3))
    return [rv_n, rv_l, rv_p, rv_c, rv_u]


def get_distributions(size):
    n = norminvgauss.rvs(1, 0, size=size)
    l = laplace.rvs(size=size, scale=1 / m.sqrt(2), loc=0)
    p = poisson.rvs(10, size=size)
    c = cauchy.rvs(size=size)
    u = uniform.rvs(size=size, loc=-m.sqrt(3), scale=2 * m.sqrt(3))
    counted_distributions = [n, l, p, c, u]
    return counted_distributions


def get_full_name(name):
    if name == 'n':
        full_name = 'Normal'
    elif name == 'p':
        full_name = 'Poisson'
    elif name == 'u':
        full_name = 'Uniform'
    elif name == 'l':
        full_name = 'Laplace'
    else:
        full_name = 'Cauchy'
    return full_name


def show_imp_f(distr, real_distr, name, size):
    a = -4
    b = 4
    full_name = get_full_name(name)

    if name == 'p':
        a = 6
        b = 14
    x_axis = np.arange(a, b, 0.01)
    d, sorted_x = get_count_vals(distr)
    ys = get_full_values_list(d, sorted_x, size, x_axis)
    plt.plot(x_axis, ys, color='red')
    plt.plot(x_axis, real_distr.cdf(x_axis), color='blue')
    plt.title(full_name + ' n = ' + str(size))
    plt.show()


def show_density_f(distr, real_density, name, size, h_coeff):
    a = -4
    b = 4
    full_name = get_full_name(name)
    if name == 'p':
        a = 6
        b = 14
        x_axis = np.arange(a, b+1, 1)
        plt.plot(x_axis, poisson.pmf(10, x_axis), lw=2)
    else:
        x_axis = np.arange(a, b, 0.01)
        plt.plot(x_axis, real_density.pdf(x_axis), color='blue')
    vals = []
    x_axis = np.arange(a, b, 0.01)
    for i in distr:
        vals.append(i)
    vals.sort()
    y = get_list_f_n(x_axis, size, vals, h_coeff)
    plt.plot(x_axis, y, color='red')
    plt.ylim(0, 1)
    plt.title(full_name + ', n = ' + str(size) + ', h = ' + str(h_coeff) + 'h_n')
    plt.show()
    pass


def get_count_vals(distr):
    d = defaultdict(int)
    vals = []
    for x in distr:
        d[x] += 1
    for val in d:
        vals.append(val)
    vals.sort()
    return d, vals


def get_full_values_list(d, sorted_vals, size, vals_list):
    sorted_y_val = []
    for point in sorted_vals:
        value = 0
        for val in sorted_vals:
            if val > point:
                break
            value += d[val]
        sorted_y_val.append(value / size)

    return_list = []
    for x_val in vals_list:
        to_append_val = 0
        i = 0
        while len(sorted_vals) > i and sorted_vals[i] <= x_val:
            to_append_val = sorted_y_val[i]
            i += 1
        return_list.append(to_append_val)
    return return_list


def h(n, array, h_coef):
    return h_coef*1.06*np.std(array)*((n+1)**(-1/5))


def k(u):
    deg = -u*u/2
    val = m.exp(deg)
    return val * (1/m.sqrt(2*m.pi))


# x_list - выборка
def f_n(point, n, x_list, h_coef):
    sum = 0
    for i in range(0, n):
        sum += k((point-x_list[i])/h(n+1, x_list, h_coef))
    return sum*(1/(n*h(n, x_list, h_coef)))


# vals_list - все х для которых считаем чтобы построить график
def get_list_f_n(vals_list, size, x_list, h_coeff):
    y_vals = []
    for point in vals_list:
        y_vals.append(f_n(point, size, x_list, h_coeff))
    return y_vals


sizes = [20, 60, 100]
names = ['n', 'l', 'p', 'c', 'u']
h_coefficients = [2]
s = get_functions()

for size in sizes:
    distributions = get_distributions(size)
    for num in range(len(distributions)):
        name = names[num]
        distribution = distributions[num]
        real_func = s[num]
        show_imp_f(distribution, real_func, name, size)
        for h_coeff in h_coefficients:
            show_density_f(distribution, real_func, name, size, h_coeff)

