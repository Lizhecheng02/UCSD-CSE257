import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

num_rows = 7
num_cols = 2

ACTIONS = {
    0: "DOWN_1",
    1: "DOWN_2",
    2: "CROSS_DOWN",
    3: "STAY"
}

valid_positions = [
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (5, 0), (5, 1),
    (6, 0), (6, 1)
]

TERMINAL_STATES = [(6, 0), (6, 1)]

position_to_state = {pos: idx for idx, pos in enumerate(valid_positions)}
state_to_position = {idx: pos for idx, pos in enumerate(valid_positions)}

num_states = len(valid_positions)
num_actions = len(ACTIONS)


class CarEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        start_positions = [(0, 0), (0, 1)]
        self.state = position_to_state[random.choice(start_positions)]
        return self.state

    def step(self, action):
        row, col = state_to_position[self.state]

        if action == 0:
            new_row, new_col = row + 1, col
        elif action == 1:
            new_row, new_col = row + 2, col
        elif action == 2:
            new_col = 1 - col
            new_row = row + 1
        elif action == 3:
            new_row, new_col = row, col

        new_position = (new_row, new_col)

        if new_position not in position_to_state:
            new_position = (row, col)

        self.state = position_to_state[new_position]

        done = new_position in TERMINAL_STATES

        reward = -1
        if done:
            reward = 10

        return self.state, reward, done


def q_learning(num_episodes=1000, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
    env = CarEnvironment()
    q_table = np.zeros((num_states, num_actions))
    d0_state = position_to_state[(2, 0)]
    d0_q_values_history = {action: [] for action in range(num_actions)}
    d0_visits = {action: 0 for action in range(num_actions)}

    for _ in tqdm(range(num_episodes), desc="Q-Learning", total=num_episodes):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < exploration_rate:
                action = random.randint(0, num_actions - 1)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)

            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + discount_factor * q_table[next_state, best_next_action]
            td_error = td_target - q_table[state, action]
            q_table[state, action] += learning_rate * td_error

            if state == d0_state:
                d0_visits[action] += 1
                d0_q_values_history[action].append((d0_visits[action], q_table[state, action]))

            state = next_state

        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    return q_table, d0_q_values_history


q_table, d0_history = q_learning(num_episodes=1000, discount_factor=0.9)

d0_state = position_to_state[(2, 0)]
d0_q_values = q_table[d0_state]

print("Q-values at D0 (position (2, 0)):")
for action, q_value in enumerate(d0_q_values):
    print(f"Action {ACTIONS[action]}: {q_value:.4f}")

print("\nOptimal Policy:")
for state in range(num_states):
    position = state_to_position[state]
    best_action = np.argmax(q_table[state])
    print(f"State {position}: {ACTIONS[best_action]}")

plt.figure(figsize=(10, 6))
for action in range(num_actions):
    if d0_history[action]:
        x_values = [entry[0] for entry in d0_history[action]]
        y_values = [entry[1] for entry in d0_history[action]]
        plt.plot(x_values, y_values, label=f"Action: {ACTIONS[action]}")

plt.xlabel("Number of Visits")
plt.ylabel("Q-Value")
plt.title("Q-Value Updates at Position D0")
plt.legend()
plt.grid(True)
plt.savefig("Figs/3-2.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
