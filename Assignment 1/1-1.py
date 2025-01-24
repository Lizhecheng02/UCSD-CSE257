import numpy as np
import matplotlib.pyplot as plt


def f1(x1, x2):
    return x1 ** 2 + 2 * x1 * x2 + 2 * x2 ** 2


x1 = np.linspace(-3, 3, 500)
x2 = np.linspace(-3, 3, 500)
X1, X2 = np.meshgrid(x1, x2)

Z = f1(X1, X2)

level_sets = [1, 3, 5]

plt.figure(figsize=(8, 6))
contour = plt.contour(X1, X2, Z, levels=level_sets, colors=["blue", "green", "red"])
plt.clabel(contour, inline=True, fontsize=10)
plt.title("Level Sets of $f_1(x_1, x_2) = x_1^2 + 2x_1x_2 + 2x_2^2$")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(alpha=0.3)
plt.savefig("Figs/1-1.pdf", bbox_inches="tight", dpi=300)
plt.show()
