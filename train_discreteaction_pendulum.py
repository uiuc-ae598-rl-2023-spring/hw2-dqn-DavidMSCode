import discreteaction_pendulum
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = discreteaction_pendulum.Pendulum()

# setup matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from iPython import display

plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create classes for replay memory
"""tuple for storing a transition"""
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        """Initialize memory"""
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition into memory"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        "Return a random memory"
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return number of stored memories"""
        return len(self.memory)

# Create the Q network class


class DQN(nn.Module):
    def __init__(self, n_obs, n_act):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obs, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_act)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.layer3(x)

#Training parameters
MAX_EPS=5000
MAX_CUDA_EPS=5000

# HYPER PARAMETERS (used as global variables)
BATCHSIZE = 128  # BATCH SIZE
GAM = 0.95  # Discount rate
EPS0 = 0.9  # explorate rate at start
EPSF = 0.05  # exploration rate at end
DECAY = 1000  # exploration decay rate
TAU = 0.005 #target network update rate
ALF = 1e-4  # learning rate of optimizer
C = 200     #frequency to reset the target network

actions = range(env.num_actions)
n_actions = len(actions)

s = env.reset()
n_obs = len(s)

Q_net = DQN(n_obs, n_actions)  #create a network for the Q function
target_net = DQN(n_obs, n_actions)  #create a target copy of the Q net
target_net.load_state_dict(Q_net.state_dict())
optimizer = optim.AdamW(Q_net.parameters(), lr=ALF, amsgrad=True)
memory = ReplayMemory(10000)

steps = 0


def select_action(state, explore=True):
    if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)
            
    global steps #get steps for exploration decay
    sample = random.random() #get a float from 0 to 1
    eps = EPSF+(EPS0-EPSF)*math.exp(-1*steps/DECAY) #calculate epsilon for current step

    if not explore or sample > eps: #if not exploring or random float larger than epsilon choose the greedy action
        with torch.no_grad():
            return Q_net(state).max(1)[1].view(1, 1) #get the action for the largest state-action value for the given state
    else:
        #otherwise explore by randomly sampling from the action space and store in a torch tensor
        return torch.tensor([random.sample(actions, 1)], device=device, dtype=torch.long)


episode_returns = []
episode_means = []


def plot_durations(episode_returns, episode_means, show_result=False, mean_num=20):
    """This function plots the learning curve as the network is training or shows the complete learning curve"""
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.plot(episode_returns)
    plt.grid()
    plt.ylim([-10, 110])
    # Take past x episode returns plot the average
    if len(episode_returns) >= mean_num and not show_result:
        episode_means.append(np.mean(episode_returns[-mean_num:-1]))
    elif not show_result:
        episode_means.append(np.sum(episode_returns)/mean_num)
    plt.plot(range(len(episode_means)), episode_means)
    plt.pause(0.0001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    return episode_means


def optimize_model():
    #Optimize model with random set of transition memories
    if len(memory) < BATCHSIZE:
        #skip if not enough memories for batch
        return
    transitions = memory.sample(BATCHSIZE)  #The sampled memories
    batch = Transition(*zip(*transitions))  #This takes the list of Transition tuples and makes a single Transition tuple of all of the transitions. i.e. batch.action gives all the actions as a tuple while transitions[i].action gives the action of the ith  transition

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool) #Get a mask transitions that do not lead to a terminal state
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)                #Tensor of states
    action_batch = torch.cat(batch.action)              #Tensor of actions
    reward_batch = torch.cat(batch.reward)              #Tensor or rewards

    state_action_values = Q_net(state_batch).gather(1, action_batch) #the Q values for the state of the current batch of memories
    next_state_values = torch.zeros(BATCHSIZE) 
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0]                        #The next state values are the max the target network for the next states

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAM) + reward_batch     #

    # Loss
    optimizer.zero_grad()
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(Q_net.parameters(), 100)
    # Optimize the model
    loss.backward()
    
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = MAX_CUDA_EPS
else:
    num_episodes = MAX_EPS

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()
    # convert to torch tensor
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)
    # init episode return (not using that name to avoid confusion with the return control)
    totalr = 0
    for t in count():
        """For each episode"""
        action = select_action(state)  # select an action by epsilon greedy
        # perform the action and return s+1, r and whether the sim is done
        s, r, done = env.step(action.item())
        # convert reward to torch tensor
        reward = torch.tensor([r], device=device)
        totalr += r  # add reward to episode returns
        if done:
            next_state = None  # next state is none if the sim has reached a terminal state
        else:
            next_state = torch.tensor(
                s, dtype=torch.float32, device=device).unsqueeze(0) #convert next state to torch tensor

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        state = next_state  # Move to the next state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if steps%C==0 and steps !=0:
            target_net_state_dict = Q_net.state_dict()
            target_net.load_state_dict(target_net_state_dict)
        
        #Soft update for target network
        # policy_net_state_dict = Q_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        # target_net.load_state_dict(target_net_state_dict)

        steps += 1 #increase step counter
        if done:
            #Episode reached terminal state, store episode return
            episode_returns.append(totalr)
            #plot graph of return each episode
            episode_means = plot_durations(episode_returns, episode_means)
            break


#Make movie
policy = lambda s: select_action(s,explore=False).item()
env.video(policy,"Trained.gif",)

print('Complete')
plot_durations(episode_returns, episode_means, show_result=True)
plt.ioff()
plt.show()
plt.savefig("temp.png")
