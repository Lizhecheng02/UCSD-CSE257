import numpy as np
import matplotlib.pyplot as plt


def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return x1 ** 2 + x2 ** 2 + np.sin(10 * x1) * np.sin(10 * x2)


def simulated_annealing(f, x_init, T0=1, alpha=0.95, iterations=100, seed=None):
    np.random.seed(seed)
    x = np.array(x_init)
    path = [x.copy()]
    T = T0

    for _ in range(iterations):
        x_new = x + np.random.normal(0, 0.1, size=x.shape)
        delta = f(x_new[0], x_new[1]) - f(x[0], x[1])
        if delta < 0 or np.exp(-delta / T) > np.random.rand():
            x = x_new
        path.append(x.copy())
        T *= alpha

    return np.array(path)


def gradient_descent(f_grad, x_init, alpha=0.02, iterations=100):
    x = np.array(x_init)
    path = [x.copy()]

    for _ in range(iterations):
        grad = np.array(f_grad(x[0], x[1]))
        x = x - alpha * grad
        path.append(x.copy())

    return np.array(path)


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
        grad = np.mean((samples - mu) * (scores[:, np.newaxis] - np.mean(scores)), axis=0)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 10:
            grad = (grad / grad_norm) * 10
        mu = mu - alpha * grad
        mu_history.append(mu.copy())

    return np.array(mu_history)


def f1_grad(x1, x2):
    return np.array([2 * x1, 2 * x2])


def f2_grad(x1, x2):
    return np.array([
        2 * x1 + 10 * np.cos(10 * x1) * np.sin(10 * x2),
        2 * x2 + 10 * np.sin(10 * x1) * np.cos(10 * x2),
    ])


def plot_results(f, f_grad, title, seeds=[0, 1, 2]):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    levels = np.linspace(0.1, 2, 20)

    _, axs = plt.subplots(1, 3, figsize=(18, 6))

    for ax, seed in zip(axs, seeds):
        ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.5)

        sa_path = simulated_annealing(f, x_init=[1, 1], seed=seed)
        ax.plot(sa_path[:, 0], sa_path[:, 1], "x-", label="SA", markersize=3)

        gd_path = gradient_descent(f_grad, x_init=[1, 1])
        ax.plot(gd_path[:, 0], gd_path[:, 1], "*-", label="GD", markersize=3)

        ce_path = cross_entropy(f, mu_init=[1, 1], sigma_init=np.eye(2), seed=seed)
        ax.plot(ce_path[:, 0], ce_path[:, 1], "o-", label="CE", markersize=3)

        sg_path = search_gradient(f, mu_init=[1, 1], sigma_init=0.1 * np.eye(2), seed=seed)
        ax.plot(sg_path[:, 0], sg_path[:, 1], "s-", label="SG", markersize=3)

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_title(f"{title} (Seed {seed})")
        ax.legend()

    plt.tight_layout()
    plt.show()


plot_results(f1, f1_grad, "Function f1")
plot_results(f2, f2_grad, "Function f2")
