import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Pendulum():
    def __init__(self, rg=None, num_actions=31):
        # Parameters that describe the physical system
        self.params = {
            'm': 1.0,   # mass
            'g': 9.8,   # acceleration of gravity
            'l': 1.0,   # length
            'b': 0.1,   # coefficient of viscous friction
        }

        # Maximum absolute angular velocity
        self.max_thetadot = 15.0

        # Maximum absolute angle to be considered "upright"
        self.max_theta_for_upright = 0.1 * np.pi

        # Maximum absolute angular velocity from which to sample initial condition
        self.max_thetadot_for_init = 5.0

        # Maximum absolute torque
        self.max_tau = 5.0

        # Time step
        self.dt = 0.1

        # Random number generator
        if rg is None:
            self.rg = np.random.default_rng()
        else:
            self.rg = rg

        # Number of states (the state space is continuous, so "number of states"
        # means the dimension of this continuous space)
        self.num_states = 2

        # Number of actions (should be odd, so that there is always action that
        # corresponds to zero torque)
        self.num_actions = num_actions

        # Time horizon
        self.max_num_steps = 100

        # Reset to initial conditions
        self.reset()

    def _x_to_s(self, x):
        return np.array([((x[0] + np.pi) % (2 * np.pi)) - np.pi, x[1]])

    def _a_to_u(self, a):
        return -self.max_tau + ((2 * self.max_tau * a) / (self.num_actions - 1))

    def _dxdt(self, x, u):
        theta_ddot =  (u - self.params['b'] * x[1] + self.params['m'] * self.params['g'] * self.params['l'] * np.sin(x[0])) / (self.params['m'] * self.params['l']**2)
        return np.array([x[1], theta_ddot])

    def set_rg(self, rg):
        self.rg = rg

    def step(self, a):
        # Verify action is in range
        if not (a in range(self.num_actions)):
            raise ValueError(f'invalid action {a}')

        # Convert a to u
        u = self._a_to_u(a)

        # Solve ODEs to find new x
        sol = scipy.integrate.solve_ivp(fun=lambda t, x: self._dxdt(x, u), t_span=[0, self.dt], y0=self.x, t_eval=[self.dt])
        self.x = sol.y[:, 0]

        # Convert x to s (same but with wrapped theta)
        self.s = self._x_to_s(self.x)

        # Get theta and thetadot
        theta = self.s[0]
        thetadot = self.s[1]

        # Compute reward
        if abs(thetadot) > self.max_thetadot:
            # If constraints are violated, then return large negative reward
            r = -100
        elif abs(theta) < self.max_theta_for_upright:
            # If pendulum is upright, then return small positive reward
            r = 1
        else:
            # Otherwise, return zero reward
            r = 0

        # Increment number of steps and check for end of episode
        self.num_steps += 1
        done = (self.num_steps >= self.max_num_steps)

        return (self.s, r, done)

    def reset(self):
        # Sample theta and thetadot
        self.x = self.rg.uniform([-np.pi, -self.max_thetadot_for_init], [np.pi, self.max_thetadot_for_init])

        # Convert x to s (same but with wrapped theta)
        self.s = self._x_to_s(self.x)

        # Reset current time (expressed as number of simulation steps taken so far) to zero
        self.num_steps = 0

        return self.s

    def video(self, policy, filename='pendulum.gif', writer='imagemagick'):
        s = self.reset()
        s_traj = [s]
        done = False
        while not done:
            (s, r, done) = self.step(policy(s))
            s_traj.append(s)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        ax.set_aspect('equal')
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        text = ax.set_title('')

        def animate(i):
            theta = s_traj[i][0]
            line.set_data([0, -np.sin(theta)], [0, np.cos(theta)])
            text.set_text(f'time = {i * self.dt:3.1f}')
            return line, text

        anim = animation.FuncAnimation(fig, animate, len(s_traj), interval=(1000 * self.dt), blit=True, repeat=False)
        anim.save(filename, writer=writer, fps=10)

        plt.close()
