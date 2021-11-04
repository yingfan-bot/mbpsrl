import numpy as np
import gym
import heapq
import argparse
import json
import os
import torch
import scipy.stats as stats

class CEM():
    def __init__(self, env, args, my_dx, my_cost, num_elites, num_trajs, alpha):
        self.env = env
        self.env_name = args.env
        self.num_elites = num_elites
        self.num_trajs = num_trajs
        self.alpha = alpha
        self.plan_hor = args.plan_hor
        self.max_iters = args.max_iters
        self.epsilon = 0.01
        self.my_dx = my_dx
        self.cost = my_cost
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
        samples = X.rvs(size=[self.num_trajs, self.soln_dim]) * np.sqrt(constrained_var) + means
        solution = samples.copy().T

        return solution, means, vars


    def hori_planning(self, cur_s):
        cur_s = cur_s.squeeze()
        '''choose elite actions from simulation trajectorys from current timestep t'''
        action_shape = len([self.env.action_space.sample()])
        init_means = np.concatenate((self.pre_means[self.action_shape:], np.zeros(self.action_shape)))

        init_vars = self.args.var * np.ones(self.action_shape * self.plan_hor)

        means = init_means
        vars = init_vars

        '''first sampling from initial distribution'''

        X = stats.truncnorm(-2, 2, loc=np.zeros_like(means), scale=np.ones_like(means))
        lb_dist, ub_dist = means - self.lb, self.ub - means
        constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
        samples = X.rvs(size=[self.num_trajs, self.soln_dim]) * np.sqrt(constrained_var) + means
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

    def get_elites(self, cur_s, sample_hori_actions):
        # compute total costs for each trajs and select the top ones
        pre_cum_hori_rewards = np.zeros([self.num_trajs, 1])
        pre_s = cur_s.numpy().copy()
        # concat all trajs started with current state
        pre_ss = [pre_s for i in range(self.num_trajs)]
        pre_ss = np.array(pre_ss)
        for t in range(self.plan_hor):
            action_s = sample_hori_actions[t * self.action_shape:(t + 1) * self.action_shape].T.copy()

            xu = np.concatenate((pre_ss.squeeze(), action_s), 1)
            new_pre_ss = self.my_dx.predict(xu)
            pre_r = self.cost.predict(torch.Tensor(xu))
            pre_ss = new_pre_ss
            pre_cum_hori_rewards += pre_r

        pre_cum_hori_rewards = np.nan_to_num(pre_cum_hori_rewards)
        elite_indices = list(
            map(pre_cum_hori_rewards.tolist().index, heapq.nlargest(self.num_elites, pre_cum_hori_rewards.tolist())))
        best_indice = pre_cum_hori_rewards.tolist().index(max(pre_cum_hori_rewards.tolist()))

        return pre_cum_hori_rewards, elite_indices, best_indice


