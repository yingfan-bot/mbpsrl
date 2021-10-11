import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math

class PendulumEnv():
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()
        self.max_steps = 200

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def _step(self, obs, u):

    #     cos_th, sin_th, thdot = obs
    #     th = math.atan(sin_th / cos_th)

    #     g = 10.
    #     m = 1.
    #     l = 1.
    #     dt = self.dt

    #     u = np.clip(u, -self.max_torque, self.max_torque)[0]
    #     self.last_u = u # for rendering
    #     costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

    #     newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
    #     newth = th + newthdot*dt + np.random.normal(loc=0, scale=0.01, size=[1])
    #     newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) + np.random.normal(loc=0, scale=0.01, size=[1]) #pylint: disable=E1111

    #     state = np.array([newth, newthdot])
    #     theta, thetadot = state
    #     new_state = np.array([np.cos(theta), np.sin(theta), thetadot])

    #     return new_state, -costs, False, {}

    def step(self, u):

        th, thdot = self.state  # th := theta
        # print('theta', th)
        # th, thdot = s # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt + np.random.normal(loc=0, scale=0.01, size=[1])
        newth = th + newthdot*dt + np.random.normal(loc=0, scale=0.01, size=[1])
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth[0], newthdot[0]])

        if self.cur_step <= self.max_steps:
            done = False
            self.cur_step += 1
        else:
            done = True

        return self._get_obs(), -costs, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        #self.state = self.np_random.uniform(low=-high, high=high).reshape(2,1)
        # self.state = np.array([-3.13648249, -0.02816067])  # the bottom
        # initial state!

        self.state = np.array([[3.14159], [-0.02816067]])  # down
        #self.state = np.array([[0], [-0.02816067]]) #up
        self.last_u = None
        self.cur_step = 0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)