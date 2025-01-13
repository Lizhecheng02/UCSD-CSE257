import numpy as np
import matplotlib.pyplot as plt

mu = np.array([1, 1])
Sigma = np.array([[1, 1], [1, 2]])
Sigma_inv = np.linalg.inv(Sigma)


def gaussian_density(x1, x2):
    x = np.array([x1, x2]) - mu
    exponent = -0.5 * np.dot(np.dot(x.T, Sigma_inv), x)
    return (1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))) * np.exp(exponent)


x1 = np.linspace(-1, 3, 500)
x2 = np.linspace(-1, 3, 500)
X1, X2 = np.meshgrid(x1, x2)

Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = gaussian_density(X1[i, j], X2[i, j])

level_sets = [0.05, 0.1, 0.15]

plt.figure(figsize=(8, 6))
contour = plt.contour(X1, X2, Z, levels=level_sets, colors=["blue", "green", "red"])
plt.clabel(contour, inline=True, fontsize=10)
plt.title("Level Sets of the Gaussian Density Function")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.scatter(mu[0], mu[1], color="black", label="Mean ($\\mu$)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
