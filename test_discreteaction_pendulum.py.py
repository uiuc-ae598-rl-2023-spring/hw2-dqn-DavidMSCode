import random
import numpy as np
import matplotlib.pyplot as plt
import discreteaction_pendulum


def main():
    # Create environment
    #
    #   By default, the action space (tau) is discretized with 31 grid points.
    #
    #   You can change the number of grid points as follows (for example):
    #
    #       env = discrete_pendulum.Pendulum(num_actions=21)
    #
    #   Note that there will only be a grid point at "0" if the number of grid
    #   points is odd.
    #
    #   How does performance vary with the number of grid points? What about
    #   computation time?
    env = discreteaction_pendulum.Pendulum()

    ######################################
    #
    #   EXAMPLE OF CREATING A VIDEO
    #

    # Define a policy that maps every state to the "zero torque" action
    policy = lambda s: env.num_actions // 2

    # Simulate an episode and save the result as an animated gif
    env.video(policy, filename='figures/test_discreteaction_pendulum.gif')

    #
    ######################################

    ######################################
    #
    #   EXAMPLE OF CREATING A PLOT
    #

    # Initialize simulation
    s = env.reset()

    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = random.randrange(env.num_actions)
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = [env._a_to_u(a) for a in data['a']]

    # Plot data and save to png file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(data['t'][:-1], data['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('figures/test_discreteaction_pendulum.png')

    #
    ######################################


if __name__ == '__main__':
    main()
