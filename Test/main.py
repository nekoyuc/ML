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
import argparse

# ... Other RL algorithm imports ...

# Environment parameters
SCREEN_X = 600
SCREEN_Y = 600
GOAL_WIDTH = 400
GOAL_HEIGHT = 250
BLOCK_WIDTH = 10
BLOCK_HEIGHT = 20
MAX_JOINTS = 20

NUM_EPISODES = 500
#MAX_STEPS_PER_EPISODE = 1000

# Hyperparameters
CRITIC_LEARNING_RATE = 0.001
ACTOR_LEARNING_RATE = 0.0005
DISCOUNT_FACTOR = 0.95
REPLAY_BUFFER_CAPACITY = 100000
EPSILON = 0.99 # Initial exploration rate
EPSILON_DECAY = 0.9995 # How quickly exploration decreases
BATCH_SIZE = 1024
GAMMA = 0.97 # Discount factor
TAU = 0.01 # Soft update rate
NOISE = 0 # Exploration noise
STABILITY_TIMEOUT_MS = 1000

# Parse "--disable-rendering" argument
parser = argparse.ArgumentParser()
parser.add_argument("--disable-rendering", action="store_true")
parser.add_argument("--experiment-name", type=str, default="experiment")
args = parser.parse_args()
# Create experiment dir if it doesn't exist
os.makedirs(args.experiment_name, exist_ok=True)

# Initialize environment, replay buffer, and model
env = TowerBuildingEnv(screen_x = SCREEN_X,
                        screen_y = SCREEN_Y,
                        goal_width = GOAL_WIDTH,
                        goal_height = GOAL_HEIGHT,
                        block_width = BLOCK_WIDTH,
                        block_height = BLOCK_HEIGHT,
                        max_joints = MAX_JOINTS)

replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

state_size = env.get_screen().shape
action_size = 3 # x, y, and rotation
hidden_layers_actor = [400, 400]
hidden_layers_critic = [400, 400]
actor = ActorNetwork(state_size, action_size, hidden_layers_actor)
critic = CriticNetwork(state_size, action_size, hidden_layers_critic)
actor = actor.to('cuda')
critic = critic.to('cuda')
target_actor = copy.deepcopy(actor)
target_critic = copy.deepcopy(critic)        


actor_optimizer = optim.Adam(actor.parameters(), lr=CRITIC_LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=ACTOR_LEARNING_RATE)

loss_history = []
score_history = []
step_index = 0
step_history = {}
critic_loss = 0.0
last_flip_time = 0
torch.autograd.set_detect_anomaly(True)
# Training loop
for episode in range(NUM_EPISODES):
    print("Episode: ", episode)
    env.reset()
    EPSILON = max(EPSILON * EPSILON_DECAY, 0.01)    
    action_index = 0
    action_history = {}
    score = 0
    state = env.get_screen()
    state = torch.tensor(np.array(state)).unsqueeze(0).float().to("cuda")
    action = torch.tensor((0.5, 0.5, 0)).float().to("cuda")
    
    while True: # Every loop places a new block
        stop = False
        start_ticks = pygame.time.get_ticks()

        # Run the simulation until the tower is stable
        while stop == False or env.calculate_stability()[1] >= 0.03:
            env.world.Step(1/10, 6, 2) #Check this step
            env.clock.tick()#10000)
            env.render()
            # Only flip if 16ms have passed since last flip
            if not args.disable_rendering:
                pygame.display.flip()
                last_flip_time = pygame.time.get_ticks()
            stop = True
            if pygame.time.get_ticks() - start_ticks >= STABILITY_TIMEOUT_MS:
                env.world.DestroyBody(env.blocks[-1].body)
                env.blocks.pop()
                break
        
        # New step stats
        score, width, height, validity = env.update_records()[1:5]
        new_state = env.get_screen()
        new_state = torch.tensor(np.array(new_state)).unsqueeze(0).float().to("cuda")
        done = env.check_done()

        # Store state from the old step before the action, and the action
        # Store the score, new state, done, validity after the action
        replay_buffer.store(state.detach(), action.detach(), score, new_state.detach(), done, validity)

        # Update state
        state = new_state
        state = state.to("cuda")

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
            # Sample a batch of experiences
            batch = replay_buffer.sample_batch(BATCH_SIZE)
            states, actions, scores, next_states, dones, validities = zip(*batch)

            # Convert to tensors
            states = torch.stack(states)
            states = states.clone().detach().float() # Shape: torce.Size([batch size, 1, state_size[0], state_size[1]])
            actions = torch.stack(actions).float() # Shape: torchSize([batch size, action_size])

            scores = np.stack(scores)
            scores = torch.tensor(scores).float()
            next_states = torch.stack(next_states)
            next_states = next_states.clone().detach().float()
            dones = torch.tensor(dones).float()
            validities = torch.tensor(validities).float()

            # Calculate critic loss
            current_q_values = critic(states, actions)
            
            with torch.no_grad():
                new_actions = target_actor(next_states)
                target_q_values = target_critic(next_states, new_actions)
                target_q = score + (GAMMA * target_q_values * (1 - done))

            critic_loss = nn.MSELoss()(current_q_values, target_q)
            # Log target q stats
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Calculate actor loss
            actor_loss = -critic(states, actor(states)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft update target networks
            _update_target(target_actor, actor, TAU)
            _update_target(target_critic, critic, TAU)

        if done:
            if type(critic_loss) != float:
                #actor_loss = 0.0
                critic_loss_record = critic_loss.cpu().item()
            else:
                critic_loss_record = critic_loss
            
            loss_history.append(critic_loss) # Store critic loss
            score_history.append(score)
            break

        # Select action
        if random.random() > EPSILON:
            action = actor(state.unsqueeze(0))[0]
            action_string = f"Unclamped Exploitation Action: {action.detach().cpu()}\n"
            action += NOISE
            action_string = action_string + f"Exploitation Action: {action.detach().cpu()}\n"
            action_history[action_index] = action_string
            step_history[step_index] = action_string
            #print(f"Exploitation Action: , {action}")
        #action = select_action(state, model, EPSILON)
        else:
            action = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            action = torch.tensor(action).float().to("cuda")
            action_string = f"No Action Clamping\nExploration Action: {action}\n"
            action_history[action_index] = action_string
            step_history[step_index] = action_string
            #print(f"Exploration Action: , {action}")
        action_index += 1
        env.step(action.detach().cpu().numpy())
        
        #score_history_string = f"Score History: , {score_history}\n"
        #print(f"Score History: , {score_history}\n")
        #print(episode_string+record_string+action_string)

    EPSILON *= EPSILON_DECAY
    # Save action history of last episode to a text file
    # Create the file if it doesn't exist

    filename = args.experiment_name + f'/action_history_episode_{env.episode}.txt'
    with open(filename, 'w') as file:
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