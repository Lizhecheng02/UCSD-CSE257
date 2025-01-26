import numpy as np
import matplotlib.pyplot as plt


def f(x1, x2):
    return x1 ** 2 - x1 * x2 + 3 * x2 ** 2 + 5


def gradient(x):
    x1, x2 = x
    grad_x1 = 2 * x1 - x2
    grad_x2 = -x1 + 6 * x2
    return np.array([grad_x1, grad_x2])


def gradient_descent(initial_point, alpha, iterations):
    x = np.array(initial_point, dtype=float)
    points = [x.copy()]

    for _ in range(iterations):
        grad = gradient(x)
        x -= alpha * grad
        points.append(x.copy())
    return np.array(points)


initial_point = [2, 2]
alpha_divergence = 0.325
iterations = 10

points_divergence = gradient_descent(initial_point, alpha_divergence, iterations)

x1_vals_div = points_divergence[:, 0]
x2_vals_div = points_divergence[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(x1_vals_div, x2_vals_div, marker="o", label="Divergent Sequence of Points")
plt.title("Steepest Gradient Descent with Divergent Stepsize")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid()
plt.xlim(x1_vals_div.min() - 1, x1_vals_div.max() + 1)
plt.ylim(x2_vals_div.min() - 1, x2_vals_div.max() + 1)
plt.savefig("Figs/5-2.pdf", bbox_inches="tight", dpi=300)
plt.show()
