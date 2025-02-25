import numpy as np
import matplotlib.pyplot as plt
import math

p1 = 0.8
p2 = 0.4


def toss(coin):
    if coin == 1:
        return 1 if np.random.rand() < p1 else 0
    else:
        return 1 if np.random.rand() < p2 else 0


T_values = np.arange(10, 1001)
J_explore1 = []
J_explore2 = []
J_epsilon = []
J_ucb = []

for T in T_values:
    N1 = math.ceil(0.2 * T)
    n1_explore = N1 // 2
    n2_explore = N1 - n1_explore
    reward_ex1 = 0
    rewards1 = []
    rewards2 = []
    for _ in range(n1_explore):
        r = toss(1)
        reward_ex1 += r
        rewards1.append(r)
    for _ in range(n2_explore):
        r = toss(2)
        reward_ex1 += r
        rewards2.append(r)
    mean1 = np.mean(rewards1) if rewards1 else 0
    mean2 = np.mean(rewards2) if rewards2 else 0
    best_coin = 1 if mean1 >= mean2 else 2
    for _ in range(T - N1):
        reward_ex1 += toss(best_coin)
    J_explore1.append(reward_ex1)

    N2 = math.ceil(0.5 * (T ** (2 / 3)) * ((np.log(T)) ** (1 / 3)))
    n1_explore = N2 // 2
    n2_explore = N2 - n1_explore
    reward_ex2 = 0
    rewards1 = []
    rewards2 = []
    for _ in range(n1_explore):
        r = toss(1)
        reward_ex2 += r
        rewards1.append(r)
    for _ in range(n2_explore):
        r = toss(2)
        reward_ex2 += r
        rewards2.append(r)
    mean1 = np.mean(rewards1) if rewards1 else 0
    mean2 = np.mean(rewards2) if rewards2 else 0
    best_coin = 1 if mean1 >= mean2 else 2
    for _ in range(T - N2):
        reward_ex2 += toss(best_coin)
    J_explore2.append(reward_ex2)

    epsilon = 0.2
    reward_eps = 0
    counts = {1: 0, 2: 0}
    sums_rewards = {1: 0, 2: 0}
    if T >= 1:
        r = toss(1)
        reward_eps += r
        counts[1] += 1
        sums_rewards[1] += r
    if T >= 2:
        r = toss(2)
        reward_eps += r
        counts[2] += 1
        sums_rewards[2] += r
    for t in range(3, T + 1):
        if np.random.rand() < epsilon:
            coin = np.random.choice([1, 2])
        else:
            avg1 = sums_rewards[1] / counts[1] if counts[1] > 0 else 0
            avg2 = sums_rewards[2] / counts[2] if counts[2] > 0 else 0
            coin = 1 if avg1 >= avg2 else 2
        r = toss(coin)
        reward_eps += r
        counts[coin] += 1
        sums_rewards[coin] += r
    J_epsilon.append(reward_eps)

    reward_ucb = 0
    counts = {1: 0, 2: 0}
    sums_rewards = {1: 0, 2: 0}
    for coin in [1, 2]:
        r = toss(coin)
        reward_ucb += r
        counts[coin] += 1
        sums_rewards[coin] += r
    for t in range(3, T + 1):
        ucb_values = {}
        for coin in [1, 2]:
            avg = sums_rewards[coin] / counts[coin]
            bonus = np.sqrt((2 * np.log(t)) / counts[coin])
            ucb_values[coin] = avg + bonus
        coin = 1 if ucb_values[1] >= ucb_values[2] else 2
        r = toss(coin)
        reward_ucb += r
        counts[coin] += 1
        sums_rewards[coin] += r
    J_ucb.append(reward_ucb)

plt.figure(figsize=(10, 6))
plt.plot(T_values, J_explore1, label="Explore-then-commit (N=ceil(0.2T))")
plt.plot(T_values, J_explore2, label="Explore-then-commit (N=ceil(0.5*T^(2/3)*(log T)^(1/3)))")
plt.plot(T_values, J_epsilon, label="ε-Greedy (ε=0.2)")
plt.plot(T_values, J_ucb, label="Upper Confidence Bound")
plt.xlabel("Number of Plays (T)")
plt.ylabel("Total Reward J(T)")
plt.title("Total Reward vs. Number of Plays for Different Strategies")
plt.legend()
plt.grid(True)
plt.savefig("Figs/6.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
