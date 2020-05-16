import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def function_to_min_modules(params, x, y):
    a0, a1 = params
    score = 0
    for i in range(len(x)):
        score += abs(a0*x[i]+a1-y[i])
    return score


def get_linear_regr_coefs(x, y):
    b0, b1 = get_mnk_coefs(x, y)
    result = opt.minimize(function_to_min_modules, [b0, b1], args=(x, y), method='SLSQP')
    coefs = result.x
    a0, a1 = coefs[0], coefs[1]
    return b0, b1, a0, a1


def get_mnk_coefs(x, y):
    b1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    b0 = np.mean(y) - b1 * np.mean(x)
    return b0, b1


def plot_linear_regr(x, y, text):
    b0, b1, a0, a1 = get_linear_regr_coefs(x, y)
    print('МНК: beta[0] = ' + str(b0) + ', beta[1] = ' + str(b1))
    print('МНМ: alpha[0] = ' + str(a0) + ', alpha[1] = ' + str(a1))

    plt.scatter(x[1:-2], y[1:-2], label='Выборка', edgecolor='blue')
    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Модель', color='red')
    plt.plot(x, x * (b1 * np.ones(len(x))) + b0 * np.ones(len(x)), label='МHK', color='pink')
    plt.plot(x, x * (a1 * np.ones(len(x))) + a0 * np.ones(len(x)), label='МHM', color='orange')
    plt.xlim([-1.8, 2])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(text)

    plt.savefig(text + '.png', format='png')
    plt.show()


x = np.arange(-1.8, 2, 0.2)
y = 2 * x + 2 * np.ones(len(x)) + np.random.normal(0, 1, size=len(x))
plot_linear_regr(x, y, 'NoPerturbations')
y[0] += 10
y[-1] -= 10
plot_linear_regr(x, y, 'Perturbations')
