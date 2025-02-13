import numpy as np
import matplotlib.pyplot as plt


def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return x1 ** 2 + x2 ** 2 + np.sin(10 * x1) * np.sin(10 * x2)


def cross_entropy(f, mu_init, sigma_init, num_samples=50, elite_frac=0.2, iterations=100, seed=None):
    np.random.seed(seed)
    mu = np.array(mu_init)
    sigma = np.array(sigma_init)
    mu_history = [mu.copy()]

    for _ in range(iterations):
        samples = np.random.multivariate_normal(mu, sigma, num_samples)
        scores = np.apply_along_axis(lambda x: f(x[0], x[1]), 1, samples)

        elite_idx = scores.argsort()[:int(elite_frac * num_samples)]
        elite_samples = samples[elite_idx]

        mu = elite_samples.mean(axis=0)
        sigma = np.cov(elite_samples, rowvar=False)
        mu_history.append(mu.copy())

    return np.array(mu_history)


def search_gradient(f, mu_init, sigma_init, alpha=0.02, num_samples=50, iterations=100, seed=None):
    np.random.seed(seed)
    mu = np.array(mu_init)
    sigma = np.array(sigma_init)
    mu_history = [mu.copy()]

    for _ in range(iterations):
        samples = np.random.multivariate_normal(mu, sigma, num_samples)
        scores = np.apply_along_axis(lambda x: f(x[0], x[1]), 1, samples)

        grad = np.mean((samples - mu) * scores[:, np.newaxis], axis=0)
        mu = mu - alpha * grad
        mu_history.append(mu.copy())

    return np.array(mu_history)


def plot_results(f, title, seeds=[0, 1, 2]):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    levels = np.linspace(0.1, 2, 20)
    _, axs = plt.subplots(1, 3, figsize=(18, 6))

    for ax, seed in zip(axs, seeds):
        ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.5)

        ce_path = cross_entropy(f, mu_init=[1, 1], sigma_init=np.eye(2), seed=seed)
        ax.plot(ce_path[:, 0], ce_path[:, 1], "o-", label="CE", markersize=3)

        sg_path = search_gradient(f, mu_init=[1, 1], sigma_init=0.1 * np.eye(2), seed=seed)
        ax.plot(sg_path[:, 0], sg_path[:, 1], "s-", label="SG", markersize=3)

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_title(f"{title} (Seed {seed})")
        ax.legend()

    plt.tight_layout()
    if "f1" in title:
        plt.savefig("Figs/1-2-1.pdf", bbox_inches="tight", dpi=300)
    elif "f2" in title:
        plt.savefig("Figs/1-2-2.pdf", bbox_inches="tight", dpi=300)
    plt.show()


plot_results(f1, "Function f1")
plot_results(f2, "Function f2")
