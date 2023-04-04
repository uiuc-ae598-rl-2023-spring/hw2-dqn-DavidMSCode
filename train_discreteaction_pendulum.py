# %%
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
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
# Global variables and types
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""tuple for storing a transition"""
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# Training parameters
MAX_EPS = 500
MAX_CUDA_EPS = 5000


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


def select_action(state, policy_net, explore=True):
    if not torch.is_tensor(state):
        state = torch.tensor(state, dtype=torch.float32,
                             device=device).unsqueeze(0)

    sample = random.random()  # get a float from 0 to 1
    # calculate epsilon for current step
    eps = EPSF+(EPS0-EPSF)*math.exp(-1*steps/DECAY)

    if not explore or sample > eps:  # if not exploring or random float larger than epsilon choose the greedy action
        with torch.no_grad():
            # get the action for the largest state-action value for the given state
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # get action space
        actions = range(len(policy_net(state)[0]))
        # otherwise explore by randomly sampling from the action space then store in a torch tensor
        return torch.tensor([random.sample(actions, 1)], device=device, dtype=torch.long)


def plot_learning_curve(episode_returns, episode_means, show_result=False, mean_num=20):
    """This function plots the learning curve as the network is training or shows the complete learning curve"""
    plt.figure(1)
    if show_result:
        plt.clf()
        plt.title(NAME+' Learning Curve')
    else:
        plt.clf()
        plt.title(NAME+' Training Run {:d}'.format(run_num))
    plt.xlabel('Episode')
    plt.ylabel('Episode Return')
    plt.plot(episode_returns)
    plt.grid(True)
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


def optimize_model(memory, Q_net, target_net, optimizer):
    # Optimize model with random set of transition memories
    if len(memory) < BATCHSIZE:
        # skip if not enough memories for batch
        return
    transitions = memory.sample(BATCHSIZE)  # The sampled memories
    # This takes the list of Transition tuples and makes a single Transition tuple of all of the transitions. i.e. batch.action gives all the actions as a tuple while transitions[i].action gives the action of the ith  transition
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)  # Get a mask transitions that do not lead to a terminal state

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)  # Tensor of states
    action_batch = torch.cat(batch.action)  # Tensor of actions
    reward_batch = torch.cat(batch.reward)  # Tensor or rewards

    # the Q values for the state of the current batch of memories
    state_action_values = Q_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCHSIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0]  # The next state values are the max the target network for the next states

    # Compute the expected Q values
    expected_state_action_values = (
        next_state_values * GAM) + reward_batch     #

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


def TrainPendulum(env,name,batchsize = 128, gamma = 0.95, starting_epsilon=0.9, ending_epsilon=0.05, epsilon_decay = 1000, LR=1e-4, target_reset_steps = 200, memory_length=10000, num_episodes=None):
    # HYPER PARAMETERS (used as global variables)
    global BATCHSIZE  # BATCH SIZE
    global GAM  # Discount rate
    global EPS0   # explorate rate at start
    global EPSF  # exploration rate at end
    global DECAY   # exploration decay rate
    global ALF  # learning rate of optimizer
    global C  # frequency to reset the target network
    global MEMSIZE #length of memory cache
    global NAME
    #set values
    BATCHSIZE = batchsize  # BATCH SIZE
    GAM = gamma  # Discount rate
    EPS0 = starting_epsilon  # explorate rate at start
    EPSF = ending_epsilon  # exploration rate at end
    DECAY = epsilon_decay  # exploration decay rate
    ALF = LR  # learning rate of optimizer
    C = target_reset_steps  # frequency to reset the target network
    MEMSIZE = memory_length
    NAME = name

    # other globals
    global steps  # number of transitions so far
    #setup matplotlib
    global is_ipython

    plt.ion()
    episode_returns = []
    episode_means = []  

    #choose training length based on hardware availability
    if torch.cuda.is_available() and not num_episodes:
        num_episodes = MAX_CUDA_EPS
    elif not num_episodes:
        num_episodes = MAX_EPS

    actions = range(env.num_actions)
    n_actions = len(actions)

    s = env.reset()
    n_obs = len(s)

    Q_net = DQN(n_obs, n_actions)  # create a network for the Q function
    target_net = DQN(n_obs, n_actions)  # create a target copy of the Q net
    target_net.load_state_dict(Q_net.state_dict())
    optimizer = optim.AdamW(Q_net.parameters(), lr=ALF, amsgrad=True)
    memory = ReplayMemory(MEMSIZE)

    # initialize steps to 0
    steps = 0

    #Start training agent
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        s = env.reset()
        # convert to torch tensor
        state = torch.tensor(s, dtype=torch.float32,
                             device=device).unsqueeze(0)
        # init episode return (not using that name to avoid confusion with the return control)
        totalr = 0
        #Start simulating episode until completion
        for t in count():
            """For each episode"""
            action = select_action(
                state, Q_net)  # select an action by epsilon greedy
            # perform the action and return s+1, r and whether the sim is done
            s, r, done = env.step(action.item())
            # convert reward to torch tensor
            reward = torch.tensor([r], device=device)
            totalr += r  # add reward to episode returns
            if done:
                next_state = None  # next state is none if the sim has reached a terminal state
            else:
                next_state = torch.tensor(
                    s, dtype=torch.float32, device=device).unsqueeze(0)  # convert next state to torch tensor

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            state = next_state  # Move to the next state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory,Q_net,target_net,optimizer)

            #Reset the target network every C steps
            if steps % C == 0:
                target_net_state_dict = Q_net.state_dict()
                target_net.load_state_dict(target_net_state_dict)

            steps += 1  # increase step counter
            if done:
                # Episode reached terminal state, store episode return
                episode_returns.append(totalr)
                # plot graph of return each episode
                episode_means = plot_learning_curve(episode_returns, episode_means)
                break
    #Turn off interactive plotting
    plt.ioff()
    return Q_net, episode_returns, episode_means

def get_average_run(env,max_runs,name="Default",color='blue',*args):
    global run_num
    nets = []
    returns = []
    means = []
    for run_num in range(max_runs):
        trained_net, episode_returns, episode_means = TrainPendulum(env,name,*args)
        nets.append(trained_net)
        returns.append(episode_returns)
        means.append(episode_means)
        print('Finished run {:d}'.format(run_num))
    means = np.array(means)
    returns = np.array(returns)
    averaged_returns = np.mean(returns,axis=0)

    return nets, returns, means, averaged_returns

def plot_returns(ax,returns,colors):
    for ret,color in zip(returns,colors):
        for i in range(len(ret)):
            plt.plot(ret[i],'o',color=color, alpha = 0.2,markeredgewidth=0)
    return ax
def plot_mean_returns(ax,avg_returns,colors,names):
    for avg,color,name in zip(avg_returns,colors,names):
        plt.plot(avg,color=color,label=name)
    return ax

def simulate_episode(env, policy):
    s = env.reset()
    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }
    # Simulate until episode is done
    done = False
    #run sim until episode ends
    while not done:
        a = policy(s)                        #choose the next action
        (s1, r, done) = env.step(a)                             #transition to next state
        s = s1                                                  #make next state the current state
        # Store current step in log
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    return log

def TD0ValueEstimate(env,policy,alf,discount,max_episodes):
    V = np.array()
    for i in np.arange(0,max_episodes):
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(policy[s])
            (s1, r, done) = env.step(a)                     #transition to next state
            V[s] = V[s] + alf*(r + discount*V[s1]-V[s])  #update V with TD
            s=s1
    return V
# %%
if __name__ == '__main__':
    RUN_ABLATION = True
    RUN_SINGLE_AGENT = True

    if RUN_ABLATION:
        env=discreteaction_pendulum.Pendulum()
        max_runs = 10
        max_episodes=1000
        #Hyperparameters
        """Default values for training with recall and target net"""
        batchsize = 128
        gamma = 0.95
        starting_epsilon=0.9
        ending_epsilon=0.05
        epsilon_decay = 1000
        lr=1e-4
        target_reset_steps = 200
        memory_length=10000

        name1 = 'With Recall and Target'
        color1 = 'blue'
        training_args1 = (batchsize, gamma, starting_epsilon, ending_epsilon, epsilon_decay, 
                lr, target_reset_steps, memory_length,max_episodes)
        nets1, returns1, means1, avg_returns1 = get_average_run(env,max_runs,name1,color1,*training_args1)
        """With Recall no Target"""
        target_reset_steps = 1 #reset target after every step
        name2 = "Only Recall"
        color2 = "orange"
        training_args2 = (batchsize, gamma, starting_epsilon, ending_epsilon, epsilon_decay, 
                lr, target_reset_steps, memory_length,max_episodes)
        nets2, returns2, means2, avg_returns2 =get_average_run(env,max_runs,name2,color2,*training_args2)

        """No Recall with Target"""
        target_reset_steps = 200 #reset target after every step
        memory_length = batchsize
        name3 = "Only Target"
        color3 = "green"
        training_args3 = (batchsize, gamma, starting_epsilon, ending_epsilon, epsilon_decay, 
                lr, target_reset_steps, memory_length,max_episodes)
        nets3, returns3, means3, avg_returns3=get_average_run(env,max_runs,name3,color3,*training_args3)
        """No Recall No Target"""
        target_reset_steps = 1 #reset target after every step
        memory_length = batchsize
        name4 = "No Recall or Target"
        color4 = "red"
        training_args4 = (batchsize, gamma, starting_epsilon, ending_epsilon, epsilon_decay, 
                lr, target_reset_steps, memory_length,max_episodes)
        nets4, returns4, means4, avg_returns4 =get_average_run(env,max_runs,name4,color4,*training_args4)

        fig = plt.figure('ablation',figsize=(12,6))
        fig.clf()
        plt.figure('ablation')
        ax = fig.gca()
        plt.title('Learning Curve Ablation Study')
        plt.xlabel('Episode')
        plt.ylabel('Episode Return')
        plt.grid()
        plt.ylim([-10, 110])
        # plot_returns(ax,[returns4,returns3,returns2,returns1],['blue','orange','green','red'])
        plot_mean_returns(ax,[avg_returns1,avg_returns2,avg_returns3,avg_returns4],['blue','orange','green','red'],[name1,name2,name3,name4])
        plt.legend()
        fig.savefig("figures/Ablation.png",bbox_inches='tight',dpi=300)

    if RUN_SINGLE_AGENT:
        global run_num
        run_num = 0
        max_episodes=1000
        env = discreteaction_pendulum.Pendulum()
        """Default values for training with recall and target net"""
        batchsize = 128
        gamma = 0.95
        starting_epsilon=0.9
        ending_epsilon=0.05
        epsilon_decay = 1000
        lr=1e-4
        target_reset_steps = 200
        memory_length=10000
        name = 'With Recall and Target'
        color = 'blue'
        training_args_single = (batchsize, gamma, starting_epsilon, ending_epsilon, epsilon_decay, 
                lr, target_reset_steps, memory_length,max_episodes)
        """Train an agent"""
        trained_net, episode_returns, episode_means = TrainPendulum(env,name,*training_args_single)
        """Plot learning Curve"""
        plot_learning_curve(episode_returns, episode_means,show_result=True)
        plt.savefig("figures/Learning Curve.png",bbox_inches='tight',dpi=300)
        """Plot trajectory of trained agent"""
        policy = lambda s: select_action(s,trained_net,explore=False).item()
        torque_policy = lambda s: env._a_to_u(policy(s))
        """Run episode without exploration"""
        log = simulate_episode(env,policy)
        log['s'] = np.array(log['s'])
        theta = log['s'][:, 0]
        thetadot = log['s'][:, 1]
        tau = [env._a_to_u(a) for a in log['a']]

        # Plot log and save to png file
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(log['t'], theta, label='theta')
        ax[0].plot(log['t'], thetadot, label='thetadot')
        ax[0].legend()
        ax[1].plot(log['t'][:-1], tau, label='tau')
        ax[1].legend()
        ax[2].plot(log['t'][:-1], log['r'], label='r')
        ax[2].legend()
        ax[2].set_xlabel('time step')
        plt.tight_layout()
        plt.savefig('figures/Trajectory.png')
        """Make movie with the agent"""
        env.video(policy,"figures/Trained Pendulum.gif")

        """plot policy and values"""
        print("Plotting policy and Value function")
        res = 200
        theta_max = np.pi
        theta_dot_max = env.max_thetadot
        thetas = np.linspace(-theta_max,theta_max,res)
        theta_dots = np.linspace(-theta_dot_max,theta_dot_max,res)
        Ts,TDs = np.meshgrid(thetas,theta_dots)
        policy_mesh = np.zeros((res,res))
        value_mesh = np.zeros((res,res))
        for i,theta in enumerate(thetas):
            for j,theta_dot in enumerate(theta_dots):
                s = [theta,theta_dot]
                policy_mesh[i,j] = torque_policy(s)
                s = torch.tensor(s,dtype=torch.float,device=device).unsqueeze(0)
                value_mesh[i,j] = trained_net(s).max().item()

        plt.figure(4,figsize=(8,6))
        plt.pcolor(Ts, TDs, policy_mesh)
        plt.xlabel('Theta (rads)')
        plt.ylabel('Thetadot (rads/s)')
        plt.title('Trained Agent Policy')
        cbar = plt.colorbar()
        cbar.set_label('Torque (Nm)', rotation=270)
        plt.savefig('figures/policy.png',bbox_inches='tight',dpi=300)

        plt.figure(5,figsize=(8,6))
        plt.pcolor(Ts, TDs, value_mesh)
        plt.xlabel('Theta (rads)')
        plt.ylabel('Thetadot (rads/s)')
        plt.title('Trained Agent State-Value Function')
        cbar = plt.colorbar()
        cbar.set_label('State-Value', rotation=270)
        plt.savefig('figures/value.png',bbox_inches='tight',dpi=300)
# %%
