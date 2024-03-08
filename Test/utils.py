import random
import torch
import torch.nn as nn
import numpy as np

# Hyperparameters
Kernal_Size = 3
# Outdated descrete DQN
'''
class TowerQNetWork(nn.Module): # Inherit from nn.Module
    def __init__(self, action_space):
        super(TowerQNetWork, self).__init__()
        # ... Your network layers ...

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2) # Assuming input is a 1-channel image
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2) # Assuming input is a 1-channel image

        # Calculate output size of the convolutional layers
        dummy_input = torch.zeros(1, 1, 84, 84) # Sample input
        with torch.no_grad(): # Temporarily disable gradient calculation
            conv_out = self._partial_forward(dummy_input)
        conv_out_size = conv_out.view(1, -1).size(1)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 64)
        self.fc2 = nn.Linear(64, action_space.shape[0])
    
    def _get_conv_out_size(self, shape):
        dummy_output = self.forward(torch.zeros(1, 1, *shape))
        return dummy_output.data_view(1, -1).size(1)
    
    def _partial_forward(self, x):
        # Forward pass through convolutional layers only
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x
    
    def forward(self, x):
        # ... (Your forward pass logic here) ...
        #return action_values # Or policy output, depending on your RL algorithm
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x) 
'''

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super().__init__

        '''
        # Set up input layer
        self.input_layer = nn.Linear(state_size, hidden_layers[0])

        # Set up hidden layers (adjust as needed)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            hidden_layer = nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            self.hidden_layers.append(hidden_layer)

        # Output layer (no activation here since we want raw continuous values)
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)
        '''

        # Adjust the 'in_channels' parameter if RGB images are used
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.Flatten() # Flatten the output for the fully connected layers below
        )

        # Calculate size afte convolutions
        conv_out_size = self._get_conv_out(state_size)

        # Fully connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, hidden_layers[0]),
            nn.ReLU(),
            # ...Additional hidden layers if desired ...
            nn.Linear(hidden_layers[-1], action_size)
        )

    def _get_conv_out(self, state_size):
        # Calculate output size of the convolutional layers
        dummy_input = torch.zeros(1, 1, *state_size) # batch size 1, 1 chanel, state_size
        output = self.conv_layers(dummy_input) # Pass through convolutional layers
        return output.size()[1:].numel() # Calculate flattened size

    def forward(self, state):
        x = self.conv_layers(state)
        x = self.fc_layers(x)
    
    '''
    def forward(self, state):
        # Forward pass logic (using ReLU activation for hidden layers)
        x = torch.relu(self.input_layers(state))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        action = self.output_layer(x)

        return action
    '''

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super().__init__

        # Convolutional Layers for State Processing
        self.conv_state = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 2),
            nn.ReLU(),
            # ... mroe convolutional layers if desired ...
        )

        # Process action (adjust based on integration)
        self.action_layers = nn.Sequential(
            nn.Linear(action_size, hidden_layers[0]),
            nn.ReLU(),
            # ... potentially more layers ...
        )

        # Combine state and action representations (calculate sizes accordingly)
        combined_size = self._get_combined_size(state_size, action_size)
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_size, hidden_layers[1]),
            nn.ReLU(),
            # ... Additional hidden layers if desired ...
            nn.Linear(hidden_layers[-1], 1) # Output a single Q-value
        )

    def _get_combined_size(self,state_size, action_size):
        dummy_state = torch.zeros(1, 1, *state_size)
        dummy_action = torch.zeros(1, action_size)

        x_state = self.conv_state(dummy_state)
        x_action = self.action_layers(dummy_action)

        combined_size = x_state.flatten(start_dim = 1).size(1) + x_action.size(1)
        return combined_size
        # Calcualte the combined size after state and action processing
        # ... implementation similar to _get_conv_out() ...

    def forward(self, state, action):
        # Forward pass logic (using ReLU activation for hidden layers)
        # ... implementation similar to ActorNetwork.forward() ...
        # ... but with the combined state and action representations ...

        x_state = self.conv_state(state)
        x_action = self.action_layers(action)

        x = torch.cat([x_state.flatten(start_dim = 1), x_action], dim = 1) # Concatenation
        x = self.fc_layers(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.experiences = []
        self.position = 0

    def sample_batch(self, batch_size):
        weights = np.array([e[2] + (float(e[5]) * 5) for e in self.experiences]) # Give weights to reward + 5 if valid
        probabilities = weights / weights.sum()
        print("Probabilities: ", probabilities)
        print("Type of probabilities: ", probabilities.dtype)
        print("Any NaN in probabilities: ", np.isnan(probabilities).any())
        indices = np.random.choice(len(self.experiences), batch_size, p=probabilities)
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