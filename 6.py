'''Imagine a 5x5 grid world. 
The agent starts at the bottom-left corner and needs to reach the top-right corner, 
which is the goal. 
However, there are obstacles in some cells, and the agent should learn a policy 
to navigate from the start to the goal efficiently.'''


import numpy as np

# Initialize the Q-table
num_states = 5 * 5  # 5x5 grid world
num_actions = 4  # Up, Down, Left, Right
Q_table = np.zeros((num_states, num_actions))

# Define the grid world and obstacles
grid_world = np.zeros((5, 5))
grid_world[1, 1] = -1  # Obstacle at (1, 1)
grid_world[2, 2] = -1  # Obstacle at (2, 2)
grid_world[3, 3] = -1  # Obstacle at (3, 3)
goal = (4, 4)

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration-exploitation trade-off

# Convert grid coordinates to a state index
def state_to_index(state):
    row, col = state
    return row * 5 + col

# Q-learning algorithm
def q_learning(num_episodes):
    for episode in range(num_episodes):
        state = (0, 0)  # Start at the bottom-left corner
        while state != goal:
            # Exploration-exploitation trade-off
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(Q_table[state_to_index(state)])

            # Take the chosen action and observe the next state and reward
            next_state = (state[0] + (action == 1) - (action == 0), state[1] + (action == 3) - (action == 2))
            
            # Clip the next state to ensure it stays within the bounds of the grid
            next_state = (np.clip(next_state[0], 0, 4), np.clip(next_state[1], 0, 4))
            
            reward = -1  # Small negative reward for each step

            # Update the Q-value using the Q-learning update rule
            Q_table[state_to_index(state), action] += alpha * (
                reward + gamma * np.max(Q_table[state_to_index(next_state)]) - Q_table[state_to_index(state), action]
            )

            # Move to the next state
            state = next_state

    print("Q-learning training complete.")

# Test the learned policy
def test_policy():
    state = (0, 0)
    path = [state]
    while state != goal:
        action = np.argmax(Q_table[state_to_index(state)])
        state = (state[0] + (action == 1) - (action == 0), state[1] + (action == 3) - (action == 2))
        
        # Clip the state to ensure it stays within the bounds of the grid
        state = (np.clip(state[0], 0, 4), np.clip(state[1], 0, 4))
        
        path.append(state)

    print("Optimal path:", path)

# Run Q-learning
q_learning(num_episodes=1000)

# Test the learned policy
test_policy()
