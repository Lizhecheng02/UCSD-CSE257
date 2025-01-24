import numpy as np
import matplotlib.pyplot as plt


def drop_wave(x1, x2):
    r = np.sqrt(x1 ** 2 + x2 ** 2)
    return -((1 + np.cos(12 * r)) / (0.5 * r ** 2 + 2))


def approximate_gradient_drop_wave(x, epsilon=1e-5):
    grad = np.zeros(2)
    for i in range(2):
        dx = np.zeros(2)
        dx[i] = epsilon
        grad[i] = (drop_wave(*(x + dx)) - drop_wave(*(x - dx))) / (2 * epsilon)
    return grad


initial_point = np.array([2.0, 2.0])
iterations = 30
epsilon = 0.1

three_point_values_drop_wave = []
x_three = initial_point.copy()
for _ in range(iterations):
    p = np.random.randn(2)
    p = epsilon * (p / np.linalg.norm(p))
    f_x = drop_wave(*x_three)
    f_x_p = drop_wave(*(x_three + p))
    f_x_m = drop_wave(*(x_three - p))
    if f_x_p < f_x:
        x_three += p
    elif f_x_m < f_x:
        x_three -= p
    three_point_values_drop_wave.append(drop_wave(*x_three))

gradient_values_drop_wave = []
x_grad = initial_point.copy()
for _ in range(iterations):
    grad = approximate_gradient_drop_wave(x_grad)
    grad = epsilon * (grad / np.linalg.norm(grad))
    x_grad -= grad
    gradient_values_drop_wave.append(drop_wave(*x_grad))

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, iterations + 1), three_point_values_drop_wave, label="Three-point method")
plt.xlabel("Iteration")
plt.ylabel("Function value")
plt.title("Three-point method (Drop-wave)")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, iterations + 1), gradient_values_drop_wave, label="Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Function value")
plt.title("Gradient Descent (Drop-wave)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig("Figs/2-3.pdf", bbox_inches="tight", dpi=300)
plt.show()
