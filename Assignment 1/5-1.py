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
alpha = 0.1
iterations = 10

points = gradient_descent(initial_point, alpha, iterations)

x1_vals = points[:, 0]
x2_vals = points[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(x1_vals, x2_vals, marker="o", label="Sequence of Points")
plt.title("Steepest Gradient Descent")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid()
plt.show()
