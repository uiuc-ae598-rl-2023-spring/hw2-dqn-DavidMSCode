import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env  = gym.make("CartPole-v1")

#setup matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from iPython import display

plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#create classes for replay memory
"""tuple for storing a transition"""
Transition = namedtuple('Transition',('state','action','next_state','reward'))

class  ReplayMemory(object):
    def __init__(self, capacity) -> None:
        """Initialize memory"""
        self.memory = deque([],maxlen=capacity)
    
    def push(self, *args):
        """Save a transition into memory"""
        self.memory.append(Transition(*args))

    def sample(self,batch_size):
        "Return a random memory"
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Return number of stored memories"""
        return len(self.memory)

#Create the Q network class
class DQN(nn.Module):
    def __init__(self, n_obs, n_act):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obs, 128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128, n_act)
    
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.layer3(x)


#HYPER PARAMETERS
BS = 128            #BATCH SIZW
gam = 0.95          #Discount rate
eps0 = 0.9          #explorate rate at start
epsf = 0.05         #exploration rate at end
decay = 1000        #exploration decay rate
tau = 0.005         #update rate of network
alf = 1e-4          #learning rate of optimizer

n_actions = env.action_space.n

state, info = env.reset()
n_obs = len(state)

policy_net = DQN(n_obs, n_actions)
target_net = DQN(n_obs, n_actions)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(),lr=alf,amsgrad=True)
memory = ReplayMemory(10000)

steps = 0

def select_action(state):
    global steps
    sample = random.random()
    eps = epsf+(eps0-epsf)*math.exp(-1*steps/decay)
    steps +=1
    if sample>eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        
episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BS:
        return
    transitions = memory.sample(BS)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BS)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gam) + reward_batch

    #Huber Loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 500

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.savefig("temp.png")