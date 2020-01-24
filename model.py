import torch
import torch.nn as nn
import torch.nn.functional as F

class ANetwork(nn.Module):  
    def __init__(self, state_size, action_size, seed, h1_size = 256, h2_size = 256):
        super(ANetwork, self).__init__()
        self.seed           = torch.manual_seed(seed)  # Random seed
        self.action_size    = action_size
        self.state_size     = state_size
        
        self.fc_layer1      = nn.Linear(state_size, h1_size)
        self.fc_layer2      = nn.Linear(h1_size,    h2_size)
        self.fc_layer3      = nn.Linear(h2_size,    action_size)
        

    def forward(self, state):
        x = F.relu(self.fc_layer1(state))
        x = F.relu(self.fc_layer2(x))
        return F.softmax(self.fc_layer3(x), dim = 1)
