import numpy as np
import matplotlib.pyplot as plt


def sample_mean_deviation_probability(N, epsilon, mu=0.5, num_experiments=1000):
    count_violations = 0
    for _ in range(num_experiments):
        samples = np.random.rand(N)
        xbar = np.mean(samples)
        if abs(xbar - mu) >= epsilon:
            count_violations += 1

    return count_violations / num_experiments


def hoeffding_bound(N, epsilon):
    return 2 * np.exp(-2 * N * epsilon ** 2)


if __name__ == "__main__":
    mu = 0.5
    N_values = np.arange(5, 105, 5)
    eps_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    probabilities = np.zeros((len(N_values), len(eps_values)))

    num_experiments = 1000
    for i, N in enumerate(N_values):
        for j, eps in enumerate(eps_values):
            probabilities[i, j] = sample_mean_deviation_probability(N, eps, mu=mu, num_experiments=num_experiments)

    E, Ngrid = np.meshgrid(eps_values, N_values)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(Ngrid, E, probabilities, cmap="viridis", edgecolor="none", alpha=0.8)
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\varepsilon$")
    ax.set_zlabel(r"$P(|\bar{X}_N - \mu| \geq \varepsilon)$")
    ax.set_title("Empirical Probability of Deviation (Uniform(0,1))")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig("Figs/5-3-1.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

    eps_to_compare = [0.1, 0.2]
    plt.figure(figsize=(8, 5))

    for eps in eps_to_compare:
        j = np.where(eps_values == eps)[0][0]
        empirical_prob = probabilities[:, j]
        bound_vals = [hoeffding_bound(n, eps) for n in N_values]

        plt.plot(N_values, empirical_prob, "o-", label=f"Empirical prob, eps={eps}")
        plt.plot(N_values, bound_vals, "--", label=f"Hoeffding bound, eps={eps}")

    plt.xlabel("N")
    plt.ylabel(r"Probability / Bound")
    plt.title("Comparison with Hoeffding Bound")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.savefig("Figs/5-3-2.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
