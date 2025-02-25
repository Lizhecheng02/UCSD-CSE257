import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"lines.linewidth": 1.75})
plt.rcParams.update({"lines.markersize": 6})
plt.rcParams.update({"lines.markeredgewidth": 1})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"

gamma = 0.6
S = [0, 1]
A = ["a1", "a2", "a3"]
P = np.zeros((2, 3, 2))

P[0, 0, 0] = 0.2
P[0, 0, 1] = 0.8
P[0, 1, 0] = 0.4
P[0, 1, 1] = 0.6
P[0, 2, 1] = 1.0

P[1, 0, 0] = 0.1
P[1, 0, 1] = 0.9
P[1, 2, 0] = 0.5
P[1, 2, 1] = 0.5

R = np.array([-10, 10])


def bellman_update(V):
    newV = np.zeros_like(V)
    for s in S:
        Q_values = []
        for a_idx, a in enumerate(A):
            q_val = R[s] + gamma * np.sum(P[s, a_idx] * V)
            Q_values.append(q_val)
        newV[s] = max(Q_values)
    return newV


def value_iteration(V_init, threshold=0.1, max_iters=1000):
    V = V_init.copy()
    trajectory = [V.copy()]
    for _ in range(max_iters):
        V_new = bellman_update(V)
        diff = np.max(np.abs(V_new - V))
        V = V_new
        trajectory.append(V.copy())
        if diff < threshold:
            break
    return V, trajectory


V_init_A = np.array([0.0, 0.0])
V_A_star, traj_A = value_iteration(V_init_A)

V_init_B = np.array([100.0, 100.0])
V_B_star, traj_B = value_iteration(V_init_B)

traj_A = np.array(traj_A)
traj_B = np.array(traj_B)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(traj_A[:, 0], traj_A[:, 1], marker="o")
plt.xlabel("V(s1)")
plt.ylabel("V(s2)")
plt.title("Value Iteration Trajectory from V_A = (0,0)")

plt.subplot(1, 2, 2)
plt.plot(traj_B[:, 0], traj_B[:, 1], marker="o", color="orange")
plt.xlabel("V(s1)")
plt.ylabel("V(s2)")
plt.title("Value Iteration Trajectory from V_B = (100,100)")

plt.tight_layout()
plt.savefig("Figs/1-1.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()

print("Final value from V_A:", V_A_star)
print("Final value from V_B:", V_B_star)
