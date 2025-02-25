import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({"lines.linewidth": 1.75})
plt.rcParams.update({"lines.markersize": 6})
plt.rcParams.update({"lines.markeredgewidth": 1})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"


def draw_histogram_of_means(N, p=0.7, num_trials=500, num_bins=10):
    averages = []
    for _ in range(num_trials):
        samples = np.random.binomial(1, p, size=N)
        avg = np.mean(samples)
        averages.append(avg)

    plt.hist(averages, bins=np.linspace(0, 1, num_bins + 1), alpha=0.7, edgecolor="black")
    plt.title(f"Histogram of 500 means of N={N}")
    plt.xlabel(r"$\bar{X}_N$")
    plt.ylabel("Frequency")
    plt.xlim([0, 1])


if __name__ == "__main__":
    N_values = [1, 50, 1000]
    plt.figure(figsize=(12, 4))
    for i, N in enumerate(N_values, 1):
        plt.subplot(1, 3, i)
        draw_histogram_of_means(N)
    plt.tight_layout()
    plt.savefig("Figs/5-1.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
