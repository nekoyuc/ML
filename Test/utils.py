import random
import torch
import torch.nn as nn
import numpy as np

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
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, hidden_layers[0]),
            nn.ReLU(),
            # ...Additional hidden layers if desired ...
            nn.Linear(hidden_layers[-1], action_size),
            nn.ReLU()
        )

    def _get_conv_out(self, state_size):
        # Calculate output size of the convolutional layers
        dummy_input = torch.zeros(1, 1, *state_size) # batch size 1, state_size (1, 256, 256)
        output = self.conv_layers(dummy_input) # Pass through convolutional layers
        conv_out_size = output.size()[1:].numel() # Calculate the output size
        return conv_out_size

    def forward(self, state):
        x = self.conv_layers(state)
        x = self.fc_layers(x)
        return x

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
        dummy_state = torch.zeros(1, 1, *state_size) # batch size 1, state_size (1, 256, 256)
        dummy_action = torch.zeros(1, action_size) # batch size 1, action_size

        x_state = self.conv_state(dummy_state)
        x_action = self.action_layers(dummy_action)

        combined_size = x_state.flatten(start_dim = 1).size(1) + x_action.size(1)
        return combined_size
        # Calcualte the combined size after state and action processing
        # ... implementation similar to _get_conv_out() ...

    def forward(self, state, action):
        x_state = self.conv_state(state)
        x_action = self.action_layers(torch.Tensor(action).unsqueeze(0))

        print("x_state size: ", x_state.size())
        print("x_action size: ", x_action.size())

        x_action = action.squeeze(0)

        x = torch.cat([x_state.flatten(start_dim = 1), x_action], dim = 1) # Concatenation
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
        indices = np.random.choice(len(self.experiences), batch_size, p = probabilities)
        batch = [self.experiences[index] for index in indices]
        return batch

    def store(self, state, action, reward, next_state, done, is_valid):
        experience = (state, action, reward, next_state, done, is_valid)
        if len(self.experiences) < self.capacity:
            self.experiences.append(None)
        self.experiences[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.experiences)

def _update_target(target_network, main_network, tau):
    '''
    Performs a soft update of the target network parameters using the main network parameters

    Args:
        target_network (nn.Module): The target network to update
        main_network (nn.Module): The main network from which to draw the updated parameters
        tau (float): The interpolation parameter for the update, the rate of parameter blending.
            A value of 1.0 means hard update. Usually a value between 0.01 and 0.001 is used.
    '''
    for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)