import numpy as np
import matplotlib.pyplot as plt


def drop_wave(x1, x2):
    r = np.sqrt(x1 ** 2 + x2 ** 2)
    numerator = 1 + np.cos(12 * r)
    denominator = 0.5 * (x1 ** 2 + x2 ** 2) + 2
    return -numerator / denominator


x1 = np.linspace(-2, 2, 500)
x2 = np.linspace(-2, 2, 500)
X1, X2 = np.meshgrid(x1, x2)

Z = drop_wave(X1, X2)

level_sets = [-0.8, -0.6, -0.4]

plt.figure(figsize=(8, 6))
contour = plt.contour(X1, X2, Z, levels=level_sets, colors=["blue", "green", "red"])
plt.clabel(contour, inline=True, fontsize=10)
plt.title("Level Sets of the Drop-Wave Function")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(alpha=0.3)
plt.savefig("Figs/1-3.pdf", bbox_inches="tight", dpi=300)
plt.show()
