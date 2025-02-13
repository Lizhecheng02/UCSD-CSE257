import numpy as np
import matplotlib.pyplot as plt


def f2(x1, x2):
    return x1 ** 2 + x2 ** 2 + np.sin(10 * x1) * np.sin(10 * x2)


x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f2(X1, X2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f2(x1, x2)")
ax.set_title("3D Plot of $f_2(x) = x_1^2 + x_2^2 + \sin(10x_1)\sin(10x_2)$")

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("Figs/1-1.pdf", bbox_inches="tight", dpi=300)
plt.show()
