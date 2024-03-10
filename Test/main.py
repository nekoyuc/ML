import gym
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import random
from environment import TowerBuildingEnv
from utils import ActorNetwork, CriticNetwork, ReplayBuffer, _update_target
import matplotlib.pyplot as plt
import os
import copy

# ... Other RL algorithm imports ...

# Environment parameters
SCREEN_X = 600
SCREEN_Y = 600
GOAL_WIDTH = 300
GOAL_HEIGHT = 250
GRID_SIZE = 30
MAX_JOINTS = 20

NUM_EPISODES = 200
#MAX_STEPS_PER_EPISODE = 1000

# Hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
REPLAY_BUFFER_CAPACITY = 10000
EPSILON = 1.0 # Initial exploration rate
EPSILON_DECAY = 0.995 # How quickly exploration decreases
BATCH_SIZE = 100
GAMMA = 0.99 # Discount factor
TAU = 0.01 # Soft update rate
NOISE = 0.1 # Exploration noise

# Initialize environment, replay buffer, and model
env = TowerBuildingEnv(screen_x = SCREEN_X,
                        screen_y = SCREEN_Y,
                        goal_width = GOAL_WIDTH,
                        goal_height = GOAL_HEIGHT,
                        grid_size = GRID_SIZE,
                        max_joints = MAX_JOINTS)

replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

state_size = env.get_screen().shape
print(f"State size: {state_size}")
action_size = 3 # x, y, and rotation
hidden_layers_actor = [400, 128]
hidden_layers_critic = [400, 128]
actor = ActorNetwork(state_size, action_size, hidden_layers_actor)
critic = CriticNetwork(state_size, action_size, hidden_layers_critic)
target_actor = copy.deepcopy(actor)
target_critic = copy.deepcopy(critic)

actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

loss_history = []
score_history = []

# Training loop
for episode in range(NUM_EPISODES):
    env.reset()
    is_valid = env.is_valid
    state = env.get_screen()
    state = torch.tensor(np.array(state)).unsqueeze(0).float()
    action = (random.randint(0, 600), random.randint(0, 600), random.uniform(0, 180))
    done = False

    while True: # Every loop places a new block
        stop = False

        # Run the simulation until the tower is stable
        while stop == False or env.calculate_stability()[1] >= 0.01:
            env.world.Step(1/60, 6, 2)
            env.clock.tick(10000)
            env.render()
            pygame.display.flip()
            stop = True
        
        score = env.update_records()[1] # return the latest step, score, width, and height
        new_state = env.get_screen()
        new_state = torch.tensor(np.array(new_state)).unsqueeze(0).float()
        done = env.check_done()
        print(f"Episode: {episode}, Score: {score}, Done: {done}")

        replay_buffer.store(state, action, score, new_state, done, is_valid)
        if len(replay_buffer) > BATCH_SIZE:
            batch = replay_buffer.sample_batch(BATCH_SIZE)

            # Calculate critic loss
            current_q_values = critic(state, action)
            with torch.no_grad():
                new_action = target_actor(new_state)
                target_q_values = target_critic(new_state, new_action)
                target_q = score + (GAMMA * target_q_values * (1 - done))

            critic_loss = nn.MSELoss()(current_q_values, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Calculate actor loss
            actor_loss = -critic(state, actor(state)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft update target networks
            _update_target(target_actor, actor, TAU)
            _update_target(target_critic, critic, TAU)

            loss_history.append(critic_loss.item()) # Store critic loss

        if done:
            score_history.append(score)
            break

        state = new_state
        if random.random() > EPSILON:
            action = actor(state).detach().numpy() # Convert to numpy array
            action[0:2] = torch.clamp(torch.tensor(action[0:2]), min=0, max=600)
            action[2] = torch.clamp(torch.tensor(action[2]), min=0, max=180)
            action += NOISE
        #action = select_action(state, model, EPSILON)
        else:
            action = (random.randint(0, 600), random.randint(0, 600), random.uniform(0, 180))
        env.step(action)
        EPSILON *= EPSILON_DECAY

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Loss', color=color)
ax1.plot(loss_history, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Average Score', color=color)
ax2.plot(score_history, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('DQN Training Loss and Performance')
plt.savefig('oss_and_score_plot.png')
os.system('xdg-open loss_and_score_plot.png')