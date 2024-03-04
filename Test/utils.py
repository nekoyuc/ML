import random
import torch
import torch.nn as nn
import numpy as np

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