import random
import torch
import torch.nn as nn
import numpy as np
import signal
import time
import os
import h5py

# Hyperparameters
Kernal_Size = 3
# Outdated descrete DQN

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super().__init__()

        # Adjust the 'in_channels' parameter if RGB images are used
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 4, stride = 2),
            nn.ReLU(), # 127 x 127, 4 channels
            nn.Conv2d(in_channels = 4, out_channels = 16, kernel_size = 3, stride = 2),
            nn.ReLU(), # 63 x 63, 16 channels
            nn.Flatten() # Flatten the output for the fully connected layers below
        ) # input size (1, 256, 256), output size (8, 63, 63)

        # Calculate size afte convolutions, only used when initializing the fully connected layers
        conv_out_size = self._get_conv_out(state_size)

        # Fully connected Layers
        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_layers[0]),
            nn.ReLU(),
        )

        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], action_size),
        )

        # Critic Head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[2]),
            nn.ReLU(),
            nn.Linear(hidden_layers[2], 1) # Output a single Q-value
        )

    def _get_conv_out(self, state_size):
        # Calculate output size of the convolutional layers
        dummy_input = torch.zeros(1, 1, *state_size) # batch size 1, state_size (1, 256, 256)
        output = self.conv_layers(dummy_input) # Pass through convolutional layers
        conv_out_size = output.size()[1:].numel() # Calculate the output size
        return conv_out_size
    
    def forward(self, state, action): # State shape: (batch size, 1, 256, 256)
        x = self.conv_layers(state)
        x = self.shared_fc(x)
        policy = self.actor_head(x)
        value = self.critic_head(x)
        return policy, value

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super().__init__()

        # Convolutional Layers for State Processing
        self.conv_state = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 4, out_channels = 1, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Flatten() # Flatten the output for the fully connected layers below
            # ... mroe convolutional layers if desired ...
        ) # input size (1, 1, 256, 256), output size (1, 1, 125, 125)

        # Process action (adjust based on integration)
        self.action_layers = nn.Sequential(
            nn.Linear(action_size, hidden_layers[0]),
            nn.ReLU(),
            # ... potentially more layers ...
        )

        # Combine state and action representations (calculate sizes accordingly)
        # Only used when initializing the fully connected layers
        combined_size = self._get_combined_size(state_size, action_size)

        self.fc_layers = nn.Sequential(
            nn.Linear(combined_size, hidden_layers[1]),
            nn.ReLU(),
            # ... Additional hidden layers if desired ...
            nn.Linear(hidden_layers[-1], 1) # Output a single Q-value
        )

    def _get_combined_size(self,state_size, action_size):
        dummy_state = torch.zeros(1, 1, *state_size) # batch size 1, channel size 1, state_size (256, 256)
        dummy_action = torch.zeros(1, action_size) # batch size 1, action_size
        
        x_state = self.conv_state(dummy_state) # Size: torch.Size([1, 15625])
        x_action = self.action_layers(dummy_action) # Size: torch.Size([1, 400])

        combined_size = x_state.flatten(start_dim = 1).size(1) + x_action.size(1) # 16025 with dummy input, class 'int'
        return combined_size
        # Calcualte the combined size after state and action processing
        # ... implementation similar to _get_conv_out() ...

    def forward(self, state, action):
        x_state = self.conv_state(state) # Size: torch.Size([batch size, 15625])
        x_action = self.action_layers(action) # Size: torch.Size([batch size, 400])

        x = torch.cat([x_state, x_action], dim = 1) # Concatenation
        x = self.fc_layers(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.experiences = []
        self.position = 0

    def sample_batch(self, batch_size):
        #weights = np.array([e[2] + (float(e[5]) * 5) for e in self.experiences]) # Give weights to reward + 5 if valid
        weights = np.array([e[2] + (float(e[5])) for e in self.experiences]) # Prioritize based on TD-error
        probabilities = weights / weights.sum()
        probabilities = np.maximum(probabilities, 1e-5)
        # Ensure probabilities are greater than zero and sum to 1
        probabilities = np.clip(probabilities, 0, 1)
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.experiences), batch_size, p = probabilities)
        batch = [self.experiences[index] for index in indices]
        return batch

    def store(self, state, action, reward, next_state, done, is_valid):
        experience = (state, action, reward, next_state, done, is_valid)
        if len(self.experiences) < self.capacity:
            self.experiences.append(None)
        self.experiences[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def save(self, path, episode):
        #with h5py.File(path+"/replay_buffer_"+str(episode)+".h5", "w") as f:
        with h5py.File(path+"/replay_buffer.h5", "w") as f:
            for i, (s, a, sc, ns, d, v) in enumerate(self.experiences):
                f.create_dataset(str(i)+"/state", data = s.cpu().numpy())
                f.create_dataset(str(i)+"/action", data = a.cpu().numpy())
                f.create_dataset(str(i)+"/score", data = sc)
                f.create_dataset(str(i)+"/next_state", data = ns.cpu().numpy())
                f.create_dataset(str(i)+"/done", data = d)
                f.create_dataset(str(i)+"/valid", data = v)
            f.create_dataset("position", data = self.position)
            f.create_dataset("capacity", data = self.capacity)
            print("replay buffer length: ", len(self.experiences))
            print("Saved replay buffer Lengths: ", len(f.keys()))

        print("Replay buffer saved to: ", path+"/replay_buffer.h5")

    def load(self, path):
        '''
        saved_replay_buffers = [os.path.join(path, name)
                   for name in os.listdir(path)
                   if name.endswith('.h5')]
        
        if not saved_replay_buffers:
            print("No saved replay buffers found")
            return
        
        saved_replay_buffers.sort(key = lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_replay_buffer = saved_replay_buffers[-1]
        print(f"Latest replay buffer found: {latest_replay_buffer}")
        '''

        latest_replay_buffer = path+"/replay_buffer.h5"
        if os.path.exists(latest_replay_buffer):
            with h5py.File(latest_replay_buffer, "r") as f:
                for i in range(len(f.keys())-2):
                    s = torch.tensor(f[str(i)+"/state"][:]).to("cuda")
                    a = torch.tensor(f[str(i)+"/action"][:]).to("cuda")
                    sc = float(f[str(i)+"/score"][()])
                    ns = torch.tensor(f[str(i)+"/next_state"][:]).to("cuda")
                    d = bool(f[str(i)+"/done"][()])
                    v = bool(f[str(i)+"/valid"][()])
                    self.experiences.append((s, a, sc, ns, d, v))
                self.position = f["position"][()]
                self.capacity = f["capacity"][()]
            print("Replay buffer loaded from latest")
        else:
            print("Replay buffer file not found.")

    def __len__(self):
        return len(self.experiences)

def _update_target(target_network, main_network, tau):
    '''
    Performs a soft update of the target network parameters using the main network parameters

    Args:
        target_network (nn.Module): The target network to update
        main_network (nn.Module): The main network from which to draw the updated parameters
        tau (float): The interpolation parameter for the update, the abs
        rate of parameter blending.
            A value of 1.0 means hard update. Usually a value between 0.01 and 0.001 is used.
    '''
    for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

