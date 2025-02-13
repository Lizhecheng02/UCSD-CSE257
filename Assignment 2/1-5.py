import numpy as np
import matplotlib.pyplot as plt


def f3(x):
    return np.sum(x ** 2)


def cross_entropy_experiment(f, mu_init, sigma_init, num_samples=50, elite_frac=0.2, iterations=100, seed=None):
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


def search_gradient_experiment(f, mu_init, sigma_init, alpha=0.02, num_samples=50, iterations=100, seed=None):
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


def plot_hyperparameter_experiments_extended(f, num_samples_list, elite_frac_list, alpha_list, init_variance_list, seed=0):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for num_samples in num_samples_list:
        ce_values = cross_entropy_experiment(f, mu_init=np.ones(50), sigma_init=np.eye(50), num_samples=num_samples, seed=seed)
        axs[0, 0].plot(ce_values, label=f"Samples={num_samples}")
    axs[0, 0].set_title("CE: Effect of Sample Size")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Avg Function Value")
    axs[0, 0].legend()

    for elite_frac in elite_frac_list:
        ce_values = cross_entropy_experiment(f, mu_init=np.ones(50), sigma_init=np.eye(50), elite_frac=elite_frac, seed=seed)
        axs[0, 1].plot(ce_values, label=f"Elite={elite_frac}")
    axs[0, 1].set_title("CE: Effect of Elite Fraction")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Avg Function Value")
    axs[0, 1].legend()

    for alpha in alpha_list:
        sg_values = search_gradient_experiment(f, mu_init=np.ones(50), sigma_init=0.1 * np.eye(50), alpha=alpha, seed=seed)
        axs[1, 0].plot(sg_values, label=f"Step size={alpha}")
    axs[1, 0].set_title("SG: Effect of Step Size")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Avg Function Value")
    axs[1, 0].legend()

    for variance in init_variance_list:
        sigma_init = variance * np.eye(50)
        ce_values = cross_entropy_experiment(f, mu_init=np.ones(50), sigma_init=sigma_init, seed=seed)
        axs[1, 1].plot(ce_values, label=f"Var={variance}")
    axs[1, 1].set_title("CE: Effect of Initial Variance")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Avg Function Value")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig("Figs/1-5.pdf", bbox_inches="tight", dpi=300)
    plt.show()


plot_hyperparameter_experiments_extended(
    f=f3,
    num_samples_list=[30, 50, 100],
    elite_frac_list=[0.1, 0.2, 0.5],
    alpha_list=[0.01, 0.02, 0.05],
    init_variance_list=[0.1, 1.0, 10.0]
)
