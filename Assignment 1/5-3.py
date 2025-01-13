import numpy as np
import matplotlib.pyplot as plt


def gradient(x):
    x1, x2 = x
    grad_x1 = 2 * x1 - x2
    grad_x2 = -x1 + 6 * x2
    return np.array([grad_x1, grad_x2])


def hessian():
    return np.array([[2, -1], [-1, 6]])


def newton_descent(initial_point, alpha, iterations):
    x = np.array(initial_point, dtype=float)
    points = [x.copy()]

    H_inv = np.linalg.inv(hessian())

    for _ in range(iterations):
        grad = gradient(x)
        x -= alpha * np.dot(H_inv, grad)
        points.append(x.copy())
    return np.array(points)


initial_point = [2, 2]
iterations = 10

points_newton = newton_descent(initial_point, alpha=1.0, iterations=iterations)

x1_vals_newton = points_newton[:, 0]
x2_vals_newton = points_newton[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(x1_vals_newton, x2_vals_newton, marker="o", label="Newton's Method Points")
plt.title("Newton Descent with alpha=1")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.grid()
plt.show()
