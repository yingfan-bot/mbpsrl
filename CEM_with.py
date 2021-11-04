import numpy as np
import gym
import heapq
import argparse
import json
import os
import torch

import scipy.stats as stats

class CEM():
    def __init__(self, env, args, my_dx, num_elites, num_trajs, alpha):
        self.env = env
        self.env_name = args.env
        self.num_elites = num_elites
        self.num_trajs = num_trajs
        self.alpha = alpha
        self.plan_hor = args.plan_hor
        self.max_iters = args.max_iters
        self.epsilon = 0.01
        self.my_dx = my_dx
        self.args = args
        self.ub = self.env.action_space.high[0]
        self.lb = self.env.action_space.low[0]

        self.obs_shape = self.env.observation_space.shape[0]
        self.action_shape = len(self.env.action_space.sample())
        # used for mpc update
        self.soln_dim = self.action_shape * self.plan_hor
        self.pre_means = np.zeros(self.action_shape * self.plan_hor)

    def sample_hori_actions(self, means, vars, samples, elite_indices):
        '''get mean, var of horizon'''

        new_means = samples[:, elite_indices].mean(axis=1)
        new_vars = samples[:, elite_indices].var(axis=1)

        means = self.alpha * means + (1 - self.alpha) * new_means
        vars = self.alpha * vars + (1 - self.alpha) * new_vars


        X = stats.truncnorm(-2, 2, loc=np.zeros_like(means), scale=np.ones_like(means))
        lb_dist, ub_dist = means - self.lb, self.ub - means
        constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
        samples = X.rvs(size=[self.num_trajs,self.soln_dim]) * np.sqrt(constrained_var) + means
        solution = samples.copy().T


        return solution, means, vars


    def hori_planning(self, cur_s):
        cur_s = cur_s.squeeze()
        '''choose elite actions from simulation trajectorys from current timestep t'''
        action_shape = len([self.env.action_space.sample()])

        init_means = np.concatenate((self.pre_means[self.action_shape:],np.zeros(self.action_shape)))

        init_vars = self.args.var*np.ones(self.action_shape * self.plan_hor)
        means = init_means
        vars = init_vars

        '''first sampling from initial distribution'''

        X = stats.truncnorm(-2, 2, loc=np.zeros_like(means), scale=np.ones_like(means))
        lb_dist, ub_dist = means - self.lb, self.ub - means
        constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
        samples = X.rvs(size=[self.num_trajs,self.soln_dim]) * np.sqrt(constrained_var) + means
        init_solutions = samples.copy().T

        solutions = init_solutions
        iter = 0

        while iter < self.max_iters and np.max(vars) > self.epsilon:

            pre_rewards, elite_indices, best_indice = self.get_elites(cur_s, solutions)
            solutions, means, vars = self.sample_hori_actions(means, vars, solutions, elite_indices)
            iter += 1
        # print("final cumulative rewards", pre_rewards[best_indice])

        best_action = means[0:self.action_shape]
        self.pre_means = means

        return best_action

    def get_actual_cost_cartpole(self, state):
        x = state[:,0]
        theta = state[:,2]
        up_reward = np.cos(theta)
        distance_penalty_reward = -0.01 * (x ** 2)
        return up_reward + distance_penalty_reward

    def get_actual_cost_pendulum(self, state, action):
        def angle_normalize(x):
            return (((x + np.pi) % (2 * np.pi)) - np.pi)

        y = state[:, 0]
        x = state[:, 1]
        thetadot = state[:, 2]
        reward = angle_normalize(np.arctan2(x, y)) ** 2 + .1 * (thetadot ** 2) + 0.001 * action.squeeze() ** 2
        return -reward

    def get_actual_cost_pusher(self, obs):
        to_w, og_w = 0.5, 1.25
        tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], self.env.ac_goal_pos
        goal_pos = np.repeat(goal_pos.reshape(-1, 1), obs.shape[0], axis = 1).T
        goal_pos = torch.tensor(goal_pos)
        assert isinstance(obs, torch.Tensor)
        ac = np.square(obs[:,20:27]).sum(dim=1)
        tip_obj_dist = (tip_pos - obj_pos).abs().sum(dim=1)
        obj_goal_dist = (goal_pos.float() - obj_pos).abs().sum(dim=1)


        return -(to_w * tip_obj_dist + og_w * obj_goal_dist + 0.1 * ac)


    def get_actual_cost_reacher(self, obs, acs):


        ee_pos = self.get_ee_pos(obs)
        dis = ee_pos - self.env.goal

        cost = np.sum(np.square(dis), axis=1)
        cost = cost + np.sum(0.01 * (acs ** 2), axis=1)
        return -cost

    def get_ee_pos(self, states):

        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], states[:, 4:5], states[:, 5:6], states[:, 6:]

        rot_axis = np.concatenate([np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), -np.sin(theta2)],
                                  axis=1)

        rot_perp_axis = np.concatenate([-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
        cur_end = np.concatenate([
            0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
            0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
            -0.4 * np.sin(theta2)
        ], axis=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
                rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end


    def get_elites(self, cur_s, sample_hori_actions):

        pre_cum_hori_rewards = np.zeros([self.num_trajs,1])

        pre_s = cur_s.numpy().copy()
        # concat all trajs started with current state
        pre_ss = [pre_s for i in range(self.num_trajs)]
        pre_ss = np.array(pre_ss)
        for t in range(self.plan_hor):
            action_s = sample_hori_actions[t*self.action_shape:(t+1)*self.action_shape].T.copy()

            xu = np.concatenate((pre_ss.squeeze(), action_s),1)
            new_pre_ss = self.my_dx.predict(xu)

            if self.env_name == 'CartPole-continuous':
                pre_r = self.get_actual_cost_cartpole(torch.Tensor(xu))
            elif self.env_name == 'Pendulum-v0':
                pre_r = self.get_actual_cost_pendulum(pre_ss, action_s)
            elif self.env_name == 'Pusher':
                pre_r = self.get_actual_cost_pusher(torch.Tensor(xu))
            elif self.env_name == 'Reacher':
                pre_r = self.get_actual_cost_reacher(pre_ss, action_s)

            pre_ss = new_pre_ss
            if torch.is_tensor(pre_r):
                pre_r = pre_r.detach().cpu().numpy()
            pre_cum_hori_rewards += pre_r.reshape(-1, 1)

        pre_cum_hori_rewards = np.nan_to_num(pre_cum_hori_rewards)
        elite_indices = list(map(pre_cum_hori_rewards.tolist().index, heapq.nlargest(self.num_elites, pre_cum_hori_rewards.tolist())))
        best_indice = pre_cum_hori_rewards.tolist().index(max(pre_cum_hori_rewards.tolist()))


        return pre_cum_hori_rewards, elite_indices, best_indice