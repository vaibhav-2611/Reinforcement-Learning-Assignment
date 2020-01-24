import numpy as np
import random
from collections import namedtuple, deque
from model import ANetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

BUFFER_SIZE  = int(1e4)  # replay buffer size
BATCH_SIZE 	 = 64        # minibatch size
GAMMA 		 = 0.99      # discount factor
LR 			 = 5e-4      # learning rate 
UPDATE_EVERY = 4         # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():          # who will interact with environment
    def __init__(self, state_size, action_size, seed):
        self.seed           = random.seed(seed)     # random seed
        self.time           = 0                     # Initialize time step (for updating every UPDATE_EVERY steps)
        self.state_size     = state_size            # state dimension
        self.action_size    = action_size           # action dimension
        self.policy         = ANetwork(state_size, action_size, seed).to(device)    # policy network
        self.optimizer      = optim.Adam(self.policy.parameters(), lr=LR)       	# optimizer
        self.logprobs 		= []  # using as buffer
        self.rewards  		= []  # using as buffer

    def step(self, state, action, reward, next_state, done):            # Store experience; sometimes call LEARN
        self.rewards.append(reward)
        self.time = (self.time + 1) % UPDATE_EVERY                      # update timer
        if (self.time == 0):                                            # if correct time to learn
            if len(self.rewards) > BATCH_SIZE:                          # wait till memory has BATCH_SIZE number of examples
                self.learn(GAMMA)                         		 		# learn
    
    def act(self, state, eps=0.0):   # Returns action
        state 	= torch.from_numpy(state).float().unsqueeze(0)
        probs 	= self.policy(state)	# using as buffer
        m 		= Categorical(probs) 		
        action 	= m.sample()			# select a action
        self.logprobs.append(m.log_prob(action))
        return action.item()

    def learn(self, gamma = GAMMA):	# updates weights
        returns = []
        loss =[]
        R = 0
        for r in self.rewards[::-1]:
            R = r + GAMMA*R
            returns.insert(0, R)				# collect returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)	# normalise
        i = 0        
        for log_prob, R in zip(self.logprobs, returns):	
            if(i>BATCH_SIZE):
                break
            loss.append(-log_prob * R)			# compute loss
        self.optimizer.zero_grad()				# zero out earlier grad 
        loss = torch.cat(loss).sum()
        loss.backward()							# backpropagation
        self.optimizer.step()					# update weights
        self.logprobs = []
        self.rewards = []

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.seed           = random.seed(seed)
        self.batch_size     = batch_size
        self.action_size    = action_size
        self.experience     = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory         = deque(maxlen=buffer_size)  

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done): # add (s, a, r, s', done)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = []
        for i in range(1, self.batch_size + 1):
            experiences.append(self.memory[-i])

        states      = torch.from_numpy(np.vstack([e.state       for e in experiences if e is not None])).float().to(device)
        actions     = torch.from_numpy(np.vstack([e.action      for e in experiences if e is not None])).long().to(device)
        rewards     = torch.from_numpy(np.vstack([e.reward      for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state  for e in experiences if e is not None])).float().to(device)
        dones       = torch.from_numpy(np.vstack([e.done        for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
         
        return (states, actions, rewards, next_states, dones)
