import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from environment import TowerBuildingEnv
from utils import ActorNetwork, CriticNetwork, ReplayBuffer, _update_target
import matplotlib.pyplot as plt
import os
import json
import argparse

NUM_EPISODES = 2000
#MAX_STEPS_PER_EPISODE = 1000

with open('Test_Simple/config.json', 'r') as f:
    config = json.load(f)

# Hyperparameters
CRITIC_LEARNING_RATE = config['CRITIC_LEARNING_RATE'] 
ACTOR_LEARNING_RATE = config['ACTOR_LEARNING_RATE']
DISCOUNT_FACTOR = config['DISCOUNT_FACTOR']
REPLAY_BUFFER_CAPACITY = config['REPLAY_BUFFER_CAPACITY']
EPSILON_DECAY = config['EPSILON_DECAY'] # How quickly exploration decreases
BATCH_SIZE = config['BATCH_SIZE'] # Batch size for training
GAMMA = config['GAMMA'] # Discount factor
TAU = config['TAU'] # Soft update rate
NOISE = config['NOISE'] # Exploration noise
STABILITY_TIMEOUT_MS = config['STABILITY_TIMEOUT_MS'] # Time to wait for tower to stabilize
HIDDEN_LAYERS_ACTOR = config['HIDDEN_LAYERS_ACTOR']

# Save Parameters
LOAD_CHECKPOINT = config['LOAD_CHECKPOINT']
CHECKPOINT_PATH = config['CHECKPOINT_PATH']

# Parse "--disable-rendering" argument
parser = argparse.ArgumentParser()
parser.add_argument("--disable-rendering", action="store_true")
parser.add_argument("--experiment-name", type=str, default="experiment_simple")
args = parser.parse_args()
# Create experiment dir if it doesn't exist
os.makedirs(args.experiment_name, exist_ok=True)

# Initialize environment, replay buffer, and model
env = TowerBuildingEnv()
'''
env = TowerBuildingEnv(screen_x = SCREEN_X,
                        screen_y = SCREEN_Y,
                        goal_width = GOAL_WIDTH,
                        goal_height = GOAL_HEIGHT,
                        block_width = BLOCK_WIDTH,
                        block_height = BLOCK_HEIGHT,
                        max_joints = MAX_JOINTS)
'''

replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

state_size = env.get_screen().shape #[256, 256]
action_size = 3 # x, y, and rotation

#hidden_layers_critic = [400, 400]
actor = ActorNetwork(state_size, action_size, HIDDEN_LAYERS_ACTOR)
#critic = CriticNetwork(state_size, action_size, hidden_layers_critic)
actor = actor.to('cuda')
#critic = critic.to('cuda')
#target_actor = copy.deepcopy(actor)
#target_critic = copy.deepcopy(critic)


actor_optimizer = optim.Adam(actor.parameters(), lr=CRITIC_LEARNING_RATE)
#critic_optimizer = optim.Adam(critic.parameters(), lr=ACTOR_LEARNING_RATE)

loss_history = []
score_history = []
step_index = 0
step_history = {}
#critic_loss = 0
actor_loss = 0.0
last_flip_time = 0
torch.autograd.set_detect_anomaly(True)
# Training loop

def save_model(path, episode):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'EPSILON': EPSILON,
        'episode': episode,
        'model_state_dict': actor.state_dict(),
        'optimizer_state_dict': actor_optimizer.state_dict(),
    }, os.path.join(path, f"checkpoint_episode_{episode}.pth"))

    # Save loss and score history
    np.savetxt(os.path.join(path, f"loss_history_episode_{episode}.txt"), loss_history)
    np.savetxt(os.path.join(path, f"score_history_{episode}.txt"), score_history)

    print("Saving model to: ", path)

def load_latest(path):
    checkpoints = [os.path.join(path, name)
                   for name in os.listdir(path)
                   if name.endswith('.pth')]
    loss_histories = [os.path.join(path, name)
                     for name in os.listdir(path)
                     if name.startswith('loss_history_episode')]
    score_histories = [os.path.join(path, name)
                        for name in os.listdir(path)
                        if name.startswith('score_history')]
    
    if not checkpoints:
        print("No checkpoints found")
        return None, [], []

    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = checkpoints[-1]
    print(f"Latest checkpoint found: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
        
    loss_histories.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_loss_history = loss_histories[-1]
    print(f"Latest loss history found: {latest_loss_history}")
    loss_history = np.loadtxt(latest_loss_history).tolist()
    
    score_histories.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_score_history = score_histories[-1]
    print(f"Latest score history found: {latest_score_history}\n")
    score_history = np.loadtxt(latest_score_history).tolist()
    
    return checkpoint, loss_history, score_history

if LOAD_CHECKPOINT:
    checkpoint, loss_history, score_history  = load_latest(CHECKPOINT_PATH)
    if checkpoint != None:
        EPSILON = checkpoint['EPSILON']
        print(f"EPSILON: {EPSILON}")
        actor.load_state_dict(checkpoint['model_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        replay_buffer.load(CHECKPOINT_PATH)
        print(f"Checkpoint loaded from: episode {start_episode}\n")

    else:
        EPSILON = config['EPSILON'] # Initial exploration rate
        print("EPSILON: ", EPSILON)
        start_episode = 0
        print("No checkpoint loaded")
else:
    EPSILON = config['EPSILON'] # Initial exploration rate
    print("EPSILON: ", EPSILON)
    print("Starting from scratch")
    start_episode = 0

for episode in range(start_episode+1, NUM_EPISODES):
    env.episode = episode
    print(f"Episode: {episode}")
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
        while stop == False or env.calculate_stability()[1] >= 0.1:
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

        episode_string = f"Episode: {episode}\n"
        #print("Episode: ", episode)
        #record_string = f"Height: {height:.4f}, Width: {width:.4f}, Score: {score:.4f}, Done: {done}\n"
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
            dones = torch.tensor(dones).bool()
            validities = torch.tensor(validities).bool()

            with torch.no_grad():
                # Calculate critic loss
                new_actions, current_q = actor(states, actions)
                new_actions = new_actions[0]
                target_q_values = actor(next_states, new_actions)[1]
                target_q = score + (GAMMA * target_q_values * (1 - done))

            current_q.requires_grad_(True)  # Set requires_grad to True
            target_q.requires_grad_(True)  # Set requires_grad to True

            actor_loss = nn.MSELoss()(current_q, target_q)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Log target q stats
            #critic_optimizer.zero_grad()
            #critic_loss.backward()
            #critic_optimizer.step()

            # Calculate actor loss
            # actor_loss = -critic(states, actor(states)).mean()
            # actor_optimizer.zero_grad()
            # actor_loss.backward()
            # actor_optimizer.step()

            # Soft update target networks
            # _update_target(target_actor, actor, TAU)
            # _update_target(target_critic, critic, TAU)

        if done:
            if type(actor_loss) != float:
                #actor_loss = 0.0
                actor_loss_record = actor_loss.cpu().item()
            else:
                actor_loss_record = 0.0
            
            loss_history.append(actor_loss_record) # Store critic loss
            score_history.append(score)
            break

        # Select action
        if random.random() > EPSILON:
            action = actor(state.unsqueeze(0), action)[0].flatten() # State shape: [1, 1, 256, 256], Action shape: [3]
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
        step_index += 1
        action_env = action.detach().cpu().numpy().flatten()
        env.step(action_env)
        
        score_history_string = f"Score History: , {score_history}\n"
        #print(f"Score History: , {score_history}\n")
        #print(episode_string+record_string+action_string)
        #print(episode_string+action_string)
    
    if episode % 10 == 0:
        if episode != 0:
            save_model(CHECKPOINT_PATH, episode)
    
    if episode % 100 == 0:
        if episode != 0:
            replay_buffer.save(CHECKPOINT_PATH, episode)

    EPSILON *= EPSILON_DECAY
    # Save action history of last episode to a text file
    # Create the file if it doesn't exist

    filename = args.experiment_name + f'/action_history_episode_{env.episode}.txt'
    with open(filename, 'w') as file:
        for index, action_string in action_history.items():
            file.write(f"Action {index}: {action_string}\n")

# Save step history of all episodes to a text file
with open(f'/home/yucblob/src/ML/step_history_simple.txt', 'w') as file:
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
plt.savefig('loss_and_score_simple_plot.png')
os.system('xdg-open loss_and_score_simple_plot.png')