import numpy as np
import matplotlib.pyplot as plt


def f3(x):
    return np.sum(x ** 2)


def cross_entropy_high_dim(f, mu_init, sigma_init, num_samples=50, elite_frac=0.2, iterations=100, seed=None):
    np.random.seed(seed)
    mu = np.array(mu_init)
    sigma = np.array(sigma_init)
    avg_func_values = []

    for _ in range(iterations):
        samples = np.random.multivariate_normal(mu, sigma, num_samples)
        scores = np.apply_along_axis(f, 1, samples)
        avg_func_values.append(np.mean(scores))

        elite_idx = scores.argsort()[:int(elite_frac * num_samples)]
        elite_samples = samples[elite_idx]

        mu_new = elite_samples.mean(axis=0)
        sigma_new = np.cov(elite_samples, rowvar=False)

        if np.all(np.linalg.eigvals(sigma_new) > 0):
            sigma = sigma_new
        mu = mu_new

    return avg_func_values


def search_gradient_high_dim(f, mu_init, sigma_init, alpha=0.02, num_samples=50, iterations=100, seed=None):
    np.random.seed(seed)
    mu = np.array(mu_init)
    sigma = np.array(sigma_init)
    avg_func_values = []

    for _ in range(iterations):
        samples = np.random.multivariate_normal(mu, sigma, num_samples)
        scores = np.apply_along_axis(f, 1, samples)
        avg_func_values.append(np.mean(scores))

        grad = np.mean((samples - mu) * scores[:, np.newaxis], axis=0)

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            grad = grad / grad_norm

        mu = mu - alpha * grad

    return avg_func_values


def plot_high_dim_results(f, title, seeds=[0, 1, 2]):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for ax, seed in zip(axs, seeds):
        ce_values = cross_entropy_high_dim(f, mu_init=np.ones(50), sigma_init=np.eye(50), seed=seed)
        ax.plot(ce_values, label="CE")

        sg_values = search_gradient_high_dim(f, mu_init=np.ones(50), sigma_init=0.1 * np.eye(50), seed=seed)
        ax.plot(sg_values, label="SG")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Function Value")
        ax.set_title(f"{title} (Seed {seed})")
        ax.legend()

    plt.tight_layout()
    plt.savefig("Figs/1-3.pdf", bbox_inches="tight", dpi=300)
    plt.show()


plot_high_dim_results(f3, "Function f3")
