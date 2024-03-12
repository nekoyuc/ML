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
GOAL_WIDTH = 500
GOAL_HEIGHT = 250
GRID_SIZE = 30
MAX_JOINTS = 20

NUM_EPISODES = 500
#MAX_STEPS_PER_EPISODE = 1000

# Hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
REPLAY_BUFFER_CAPACITY = 10000
EPSILON = 1.0 # Initial exploration rate
EPSILON_DECAY = 0.996 # How quickly exploration decreases
BATCH_SIZE = 100
GAMMA = 0.99 # Discount factor
TAU = 0.01 # Soft update rate
NOISE = 0 # Exploration noise

# Initialize environment, replay buffer, and model
env = TowerBuildingEnv(screen_x = SCREEN_X,
                        screen_y = SCREEN_Y,
                        goal_width = GOAL_WIDTH,
                        goal_height = GOAL_HEIGHT,
                        grid_size = GRID_SIZE,
                        max_joints = MAX_JOINTS)

replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

state_size = env.get_screen().shape
action_size = 3 # x, y, and rotation
hidden_layers_actor = [400, 400]
hidden_layers_critic = [400, 400]
actor = ActorNetwork(state_size, action_size, hidden_layers_actor)
critic = CriticNetwork(state_size, action_size, hidden_layers_critic)
target_actor = copy.deepcopy(actor)
target_critic = copy.deepcopy(critic)

actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

loss_history = []
score_history = []
step_index = 0
step_history = {}
critic_loss = 0

# Training loop
for episode in range(NUM_EPISODES):
    env.reset()
    EPSILON = max(EPSILON * EPSILON_DECAY, 0.01)    
    action_index = 0
    action_history = {}
    score = 0
    state = env.get_screen()
    state = torch.tensor(np.array(state)).unsqueeze(0).float()
    action = (0.5, 0.5, 0)
    
    while True: # Every loop places a new block
        stop = False

        # Run the simulation until the tower is stable
        while stop == False or env.calculate_stability()[1] >= 0.01:
            env.world.Step(1/60, 6, 2)
            env.clock.tick(10000)
            env.render()
            pygame.display.flip()
            stop = True
        
        # New step stats
        score, width, height, validity = env.update_records()[1:5]
        new_state = env.get_screen()
        new_state = torch.tensor(np.array(new_state)).unsqueeze(0).float()
        done = env.check_done()

        # Store state from the old step before the action, and the action
        # Store the score, new state, done, validity after the action
        replay_buffer.store(state, action, score, new_state, done, validity)

        # Update state
        state = new_state

        # Debugging: Save state to a local txt file
        '''
        state_numpy = state.detach().numpy()
        state_numpy = state_numpy.squeeze()
        print("State shape: ", state_numpy.shape)
        # Save state_numpy to a local txt file
        np.savetxt(f'state_numpy_{episode}_{env.image_index}.txt', state_numpy)
        '''

        episode_string = f"Episode: {episode}\n"
        #print("Episode: ", episode)
        record_string = f"Height: {height:.4f}, Width: {width:.4f}, Score: {score:.4f}, Done: {done}\n"
        #print(f"Height: {height:.4f}, Width: {width:.4f}, Score: {score:.4f}, Done: {done}")

        if len(replay_buffer) > BATCH_SIZE:
            batch = replay_buffer.sample_batch(BATCH_SIZE)

            # Calculate critic loss
            current_q_values = critic(state, action)
            with torch.no_grad():
                new_action = target_actor(new_state)[0]
                target_q_values = target_critic(new_state, new_action)
                target_q = score + (GAMMA * target_q_values * (1 - done))

            critic_loss = nn.MSELoss()(current_q_values, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Calculate actor loss
            actor_loss = -critic(state, actor(state)[0]).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft update target networks
            _update_target(target_actor, actor, TAU)
            _update_target(target_critic, critic, TAU)

        if done:
            loss_history.append(critic_loss.item()) # Store critic loss
            score_history.append(score)
            break

        # Select action
        if random.random() > EPSILON:
            action = actor(state)[0]
            action = action.detach().numpy() # Convert to numpy array
            action_string = f"Unclamped Exploitation Action: {action}\n"
            action += NOISE
            action_string = action_string + f"Exploitation Action: {action}\n"
            action_history[action_index] = action_string
            step_history[step_index] = action_string
            #print(f"Exploitation Action: , {action}")
        #action = select_action(state, model, EPSILON)
        else:
            action = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            action_string = f"No Action Clamping\nExploration Action: {action}\n"
            action_history[action_index] = action_string
            step_history[step_index] = action_string
            #print(f"Exploration Action: , {action}")
        action_index += 1
        env.step(action)
        
        #score_history_string = f"Score History: , {score_history}\n"
        #print(f"Score History: , {score_history}\n")
        print(episode_string+record_string+action_string)

    EPSILON *= EPSILON_DECAY
    # Save action history of last episode to a text file
    with open(f'/home/yucblob/src/ML/action_history_episode_{env.episode}.txt', 'w') as file:
        for index, action_string in action_history.items():
            file.write(f"Action {index}: {action_string}\n")
    env.episode += 1

# Save step history of all episodes to a text file
with open(f'/home/yucblob/src/ML/step_history.txt', 'w') as file:
    for index, action_string in step_history.items():
        file.write(f"Step {index}: {action_string}\n")

fig, ax1 = plt.subplots(figsize=(16, 9))

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
plt.savefig('loss_and_score_plot.png')
os.system('xdg-open loss_and_score_plot.png')