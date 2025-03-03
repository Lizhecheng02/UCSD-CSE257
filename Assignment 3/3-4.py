import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

num_rows = 7
num_cols = 2

CAR_ACTIONS = {
    0: "DOWN_1",
    1: "DOWN_2",
    2: "CROSS_DOWN",
    3: "STAY"
}

D0_POS = (2, 0)
L_ROW = 3
R_ROW = 4

PARKING_SPOTS = [
    (6, 0),
    (6, 1)
]

valid_car_positions = [
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (5, 0), (5, 1),
    (6, 0), (6, 1)
]


class CarPedestrianMDP:
    def __init__(self, collision_penalty=-100):
        self.collision_penalty = collision_penalty
        self.reset()

    def reset(self):
        start_positions = [(0, 0), (0, 1)]
        self.car_pos = random.choice(start_positions)

        self.l_ped_pos = (L_ROW, random.randint(0, 7))
        self.r_ped_pos = (R_ROW, random.randint(0, 7))

        self.state = (self.car_pos, self.l_ped_pos, self.r_ped_pos)

        return self.state

    def _move_pedestrians(self):
        l_move = np.random.choice([1, 2], p=[0.5, 0.5])
        l_new_col = (self.l_ped_pos[1] - l_move) % 8
        self.l_ped_pos = (L_ROW, l_new_col)

        r_move = np.random.choice([1, 2], p=[0.5, 0.5])
        r_new_col = (self.r_ped_pos[1] + r_move) % 8
        self.r_ped_pos = (R_ROW, r_new_col)

    def _check_collision(self, car_row, car_col, prev_car_row=None, prev_car_col=None):
        if car_row == L_ROW:
            car_lane_to_full_grid = {0: 3, 1: 4}
            if car_col in car_lane_to_full_grid:
                car_full_col = car_lane_to_full_grid[car_col]
                if self.l_ped_pos[1] == car_full_col:
                    return True

        if car_row == R_ROW:
            car_lane_to_full_grid = {0: 3, 1: 4}
            if car_col in car_lane_to_full_grid:
                car_full_col = car_lane_to_full_grid[car_col]
                if self.r_ped_pos[1] == car_full_col:
                    return True

        if prev_car_row is not None and prev_car_col is not None:
            if car_row - prev_car_row == 2:
                middle_row = prev_car_row + 1
                middle_col = prev_car_col

                if middle_row == L_ROW:
                    car_lane_to_full_grid = {0: 3, 1: 4}
                    if middle_col in car_lane_to_full_grid:
                        car_full_col = car_lane_to_full_grid[middle_col]
                        if self.l_ped_pos[1] == car_full_col:
                            return True

                if middle_row == R_ROW:
                    car_lane_to_full_grid = {0: 3, 1: 4}
                    if middle_col in car_lane_to_full_grid:
                        car_full_col = car_lane_to_full_grid[middle_col]
                        if self.r_ped_pos[1] == car_full_col:
                            return True

        return False

    def step(self, action):
        prev_row, prev_col = self.car_pos

        if action == 0:
            new_row, new_col = prev_row + 1, prev_col
        elif action == 1:
            new_row, new_col = prev_row + 2, prev_col
        elif action == 2:
            new_col = 1 - prev_col
            new_row = prev_row + 1
        elif action == 3:
            new_row, new_col = prev_row, prev_col

        new_position = (new_row, new_col)
        if new_position not in valid_car_positions:
            new_position = (prev_row, prev_col)

        self._move_pedestrians()

        self.car_pos = new_position

        collision = self._check_collision(self.car_pos[0], self.car_pos[1], prev_row, prev_col)

        self.state = (self.car_pos, self.l_ped_pos, self.r_ped_pos)

        done = False
        reward = -20

        if collision:
            reward = self.collision_penalty
            done = True
        elif self.car_pos in PARKING_SPOTS:
            reward = 100
            done = True

        return self.state, reward, done


def q_learning(num_episodes=5000, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
    env = CarPedestrianMDP(collision_penalty=-100)

    q_table = {}

    d0_q_values_history = {action: [] for action in range(len(CAR_ACTIONS))}
    d0_visits = {action: 0 for action in range(len(CAR_ACTIONS))}

    for _ in tqdm(range(num_episodes), desc="Q-Learning", total=num_episodes):
        state = env.reset()
        done = False

        while not done:
            state_key = str(state)

            if state_key not in q_table:
                q_table[state_key] = np.zeros(len(CAR_ACTIONS))

            if random.uniform(0, 1) < exploration_rate:
                action = random.randint(0, len(CAR_ACTIONS) - 1)
            else:
                action = np.argmax(q_table[state_key])

            next_state, reward, done = env.step(action)
            next_state_key = str(next_state)

            if next_state_key not in q_table:
                q_table[next_state_key] = np.zeros(len(CAR_ACTIONS))

            best_next_action = np.argmax(q_table[next_state_key])
            td_target = reward + discount_factor * q_table[next_state_key][best_next_action]
            td_error = td_target - q_table[state_key][action]
            q_table[state_key][action] += learning_rate * td_error

            if state[0] == D0_POS:
                d0_visits[action] += 1
                d0_q_values_history[action].append((d0_visits[action], q_table[state_key][action]))

            state = next_state

        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    return q_table, d0_q_values_history


q_table, d0_history = q_learning(num_episodes=10000, discount_factor=0.9)

d0_states = {}
for state_key in q_table:
    state = eval(state_key)
    if state[0] == D0_POS:
        ped_positions = (state[1], state[2])
        best_action = np.argmax(q_table[state_key])
        d0_states[ped_positions] = (best_action, q_table[state_key])

print(f"Sample of learned policies at D0 (position {D0_POS}) with different pedestrian positions:")
sample_count = 0
for ped_positions, (best_action, q_values) in list(d0_states.items())[:5]:
    print(f"Pedestrians at {ped_positions}:")
    print(f"  Best action: {CAR_ACTIONS[best_action]}")
    print(f"  Q-values: {q_values}")
    sample_count += 1

if sample_count == 0:
    print("No D0 states were visited during training.")

plt.figure(figsize=(10, 6))
for action in range(len(CAR_ACTIONS)):
    if d0_history[action]:
        x_values = [entry[0] for entry in d0_history[action]]
        y_values = [entry[1] for entry in d0_history[action]]
        plt.plot(x_values, y_values, label=f"Action: {CAR_ACTIONS[action]}")

plt.xlabel("Number of Visits")
plt.ylabel("Q-Value")
plt.title("Q-Value Updates at Position D0 (Averaged across pedestrian positions)")
plt.legend()
plt.grid(True)
plt.savefig("Figs/3-4.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
