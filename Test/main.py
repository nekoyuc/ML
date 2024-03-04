import gym
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import random
from environment import TowerBuildingEnv
from utils import TowerQNetWork, ReplayBuffer

# ... Other RL algorithm imports ...

# Environment parameters
SCREEN_X = 600
SCREEN_Y = 600
GOAL_WIDTH = 300
GOAL_HEIGHT = 250
GRID_SIZE = 30
MAX_JOINTS = 20

NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000

# Hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
REPLAY_BUFFER_CAPACITY = 10000
EPSILON = 1.0 # Initial exploration rate
EPSILON_DECAY = 0.995 # How quickly exploration decreases
BATCH_SIZE = 100

# Core training logic
def update_dqn(model, optimizer, batch):
    states, actions, rewards, next_states, dones, _ = zip(*batch) # Ignore is_valid_close for now

    states = torch.tensor(states).float()
    actions = torch.tensor(actions).long() # Assuming action is discrete for now
    rewards = torch.tensor(rewards).float()
    next_states = torch.tensor(next_states).float()
    dones = torch.tensor(dones).bool()

    # Current Q values
    current_q_values = model(states)
    current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q values
    with torch.no_grad():
        max_next_q_values = model(next_states).max(1)[0]
    target_q_values = rewards + (DISCOUNT_FACTOR * max_next_q_values * (1 - dones))

    # Loss
    loss = nn.functional.mse_loss(current_q_values, target_q_values)

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def select_action(state, model, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            action_values = model(state)
            action = action_values.argmax().item()
    return action

# Initialize environment, replay buffer, and model
env = TowerBuildingEnv(screen_x = SCREEN_X,
                        screen_y = SCREEN_Y,
                        goal_width = GOAL_WIDTH,
                        goal_height = GOAL_HEIGHT,
                        grid_size = GRID_SIZE,
                        max_joints = MAX_JOINTS)
replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
model = TowerQNetWork()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for episode in range(NUM_EPISODES):
    env.reset()
    is_valid = env.is_valid
    state = env.get_screen()
    state = torch.tensor(np.array(state)).unsqueeze(0).float()
    done = False

    while True: # Every loop places a new block
        stop = False

        # Run the simulation until the tower is stable
        while stop == False or env.calculate_stability()[1] >= 0.01:
            env.world.Step(1/60, 6, 2)
            env.clock.tick(5000)
            env.render()
            pygame.display.flip()
            stop = True
        
        score = env.update_records()[1] # return the latest step, score, width, and height
        new_state = env.get_screen()
        new_state = torch.tensor(np.array(new_state)).unsqueeze(0).float()
        done = env.check_done()

        replay_buffer.store(state, action, score, new_state, done, is_valid)
        if len(replay_buffer) > BATCH_SIZE:
            batch = replay_buffer.sample_batch(BATCH_SIZE)
            loss = update_dqn(model, optimizer, batch)

        if done:
            break

        state = new_state
        action = select_action(state, model, EPSILON)
        is_valid = env.step(action)