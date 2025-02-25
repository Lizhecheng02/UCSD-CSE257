import numpy as np
import matplotlib.pyplot as plt


def empirical_deviation_probability(N, mu=0.7, epsilon=0.1, num_trials=10):
    count_violations = 0
    for _ in range(num_trials):
        samples = np.random.binomial(1, 0.7, size=N)
        xbar = np.mean(samples)
        if abs(xbar - mu) >= epsilon:
            count_violations += 1
    return count_violations / num_trials


def hoeffding_bound(N, epsilon=0.1):
    return 2 * np.exp(-2 * N * (epsilon ** 2))


if __name__ == "__main__":
    p = 0.7
    mu = p
    epsilon = 0.1
    max_N = 100
    num_trials = 10

    N_values = np.arange(1, max_N + 1)
    empirical_probs = []
    hoeffding_vals = []

    for N in N_values:
        emp_freq = empirical_deviation_probability(N, mu=mu, epsilon=epsilon, num_trials=num_trials)
        empirical_probs.append(emp_freq)
        hoeffding_vals.append(hoeffding_bound(N, epsilon=epsilon))

    plt.figure(figsize=(8, 5))
    plt.plot(N_values, empirical_probs, "bo-", label="Empirical Probability")
    plt.plot(N_values, hoeffding_vals, "r--", label="Hoeffding Bound")
    plt.xlabel("N")
    plt.ylabel(r"$P(|\bar{X}_N - \mu| \geq \varepsilon)$")
    plt.title(r"Empirical Deviation Probability vs. Hoeffding Bound ($\varepsilon=0.1$)")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.savefig("Figs/5-2.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
