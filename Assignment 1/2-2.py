import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x ** 2 + 4 * y ** 2


def gradient_f(x, y):
    return np.array([2 * x, 8 * y])


epsilon = 0.1
initial_point = np.array([1.0, 1.0])
iterations = 30

three_point_values = []
x_three = initial_point.copy()
for _ in range(iterations):
    p = np.random.randn(2)
    p = epsilon * (p / np.linalg.norm(p))
    f_x = f(*x_three)
    f_x_p = f(*(x_three + p))
    f_x_m = f(*(x_three - p))
    if f_x_p < f_x:
        x_three += p
    elif f_x_m < f_x:
        x_three -= p
    three_point_values.append(f(*x_three))

gradient_values = []
x_grad = initial_point.copy()
for _ in range(iterations):
    grad = gradient_f(*x_grad)
    grad = epsilon * (grad / np.linalg.norm(grad))
    x_grad -= grad
    gradient_values.append(f(*x_grad))

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, iterations + 1), three_point_values, label="Three-point method")
plt.xlabel("Iteration")
plt.ylabel("Function value")
plt.title("Three-point method")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, iterations + 1), gradient_values, label="Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Function value")
plt.title("Gradient Descent")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
