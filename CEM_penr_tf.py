import numpy as np
import gym
import heapq
import argparse
import json
import os
from pendulum_gym import PendulumEnv
from cartpole_continuous import ContinuousCartPoleEnv
import torch
from tf_models.constructor import construct_model
from NB_dx_tf import neural_bays_dx_tf

import pickle
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class CEM():
    def __init__(self, env, args, my_dx, num_elites, num_trajs, alpha, device):
        self.env = env
        self.env_name = args.env
        self.num_elites = num_elites
        self.num_trajs = num_trajs
        self.alpha = alpha
        self.plan_hor = args.plan_hor
        self.max_iters = args.max_iters
        self.epsilon = 0.01
        self.my_dx = my_dx
        # self.cost = my_cost
        self.device = device

        if 'CartPole-v0' in self.env_name:
            self.simulate_env = CartPoleEnv()
        elif 'Pendulum-v0' in self.env_name:
            self.simulate_env = PendulumEnv()
            self.ub = self.env.action_space.high[0]
            self.lb = self.env.action_space.low[0]
        elif 'CartPole-continuous' in self.env_name:
            self.simulate_env = ContinuousCartPoleEnv()
            self.ub = self.env.max_action
            self.lb = self.env.min_action
        self.obs_shape = self.env.observation_space.shape[0]
        self.action_shape = len([self.env.action_space.sample()])
        # used for mpc update
        self.soln_dim = self.action_shape * self.plan_hor
        self.pre_means = np.zeros(self.action_shape * self.plan_hor)

    def sample_hori_actions(self, means, vars, samples, elite_indices):
        '''get mean, var of horizon'''

        new_means = samples[:, elite_indices].mean(axis=1)
        new_vars = samples[:, elite_indices].std(axis=1)

        means = self.alpha * means + (1 - self.alpha) * new_means
        vars = self.alpha * vars + (1 - self.alpha) * new_vars

        if 'CartPole-v0' not in self.env_name:
            solution = np.array([np.clip(np.random.normal(means[t], vars[t], self.num_trajs), self.lb, self.ub) for t in
                                 range(self.plan_hor * self.action_shape)])
        else:  # discrete action space
            solution = np.array(
                [np.random.normal(means[t], vars[t], self.num_trajs) for t in range(self.plan_hor * self.action_shape)])

        return solution, means, vars

    # def hori_planning(self, cur_s):
    def hori_planning(self, cur_s):
        cur_s = cur_s.squeeze()
        '''choose elite actions from simulation trajectorys from current timestep t'''
        # action_shape = len([self.env.action_space.sample()])
        # init_means = [np.zeros(action_shape) for t in range(plan_hor)]  # for multiple dimenstions of action
        # init_vars = [np.eye(action_shape) for t in range(plan_hor)]
        # init_means = np.zeros(self.action_shape * self.plan_hor)
        # update means
        init_means = np.concatenate((self.pre_means[self.action_shape:], np.zeros(self.action_shape)))

        init_vars = 3*np.ones(self.action_shape * self.plan_hor)
        means = init_means
        vars = init_vars

        '''first sampling from initial distribution'''
        if 'CartPole-v0' not in self.env_name:
            init_solutions = np.array(
                [np.clip(np.random.normal(means[t], vars[t], self.num_trajs), self.lb, self.ub) for t in
                 range(self.plan_hor * self.action_shape)])
        else:
            init_solutions = np.array(
                [np.random.normal(means[t], vars[t], self.num_trajs) for t in range(self.plan_hor * self.action_shape)])

        solutions = init_solutions
        iter = 0

        while iter < self.max_iters and np.max(vars) > self.epsilon:
            _, elite_indices, best_indice = self.get_elites(cur_s, solutions)
            solutions, means, vars = self.sample_hori_actions(means, vars, solutions, elite_indices)
            iter += 1

        # best_action = solutions[:, best_indice][0:self.action_shape]
        # print('cem iters:',iter)
        best_action = means[0:self.action_shape]
        self.pre_means = means
        # print('best_indice', best_indice)
        # print('choose solutions', solutions[:, best_indice].shape)
        # print('best action', best_action)
        return best_action

    def get_actual_cost(self, state, action):
        def angle_normalize(x):
            return (((x + np.pi) % (2 * np.pi)) - np.pi)

        y = state[:, 0]
        x = state[:, 1]
        thetadot = state[:, 2]
        reward = angle_normalize(np.arctan2(x, y)) ** 2 + .1 * (thetadot ** 2) + 0.001 * action.squeeze() ** 2
        return -reward

    def get_elites(self, cur_s, sample_hori_actions):
        # compute total costs for each trajs and select the top ones
        # cum_hori_rewards = np.zeros(self.num_trajs)
        eps_length = np.zeros(self.num_trajs)
        pre_cum_hori_rewards = np.zeros([self.num_trajs, 1])
        # for k in range(self.num_trajs):
        pre_s = cur_s.numpy()
        # concat all trajs started with current state
        pre_ss = [pre_s for i in range(self.num_trajs)]
        pre_ss = np.array(pre_ss)
        for t in range(self.plan_hor):
            action_s = sample_hori_actions[t * self.action_shape:(t + 1) * self.action_shape].T
            # TODO: maybe need to transpose the action here
            # if 'CartPole-v0' in self.env_name:  # discrete action space for CartPole-v0
            #     action_s = 1 if action_s >= 0 else 0
            # elif 'Pendulum-v0' in self.env_name:
            #     action_s = np.array([action])
            # TODO: discrete case
            # s, r, done, _ = self.simulate_env._step(s, action)
            # # simulate_env._render()
            # cum_hori_rewards[k] += r
            # eps_length[k] += 1
            # if done:
            #     break
            # predict next state

            # xu = torch.cat((torch.Tensor(pre_ss).squeeze(), torch.Tensor(action_s)), 1)

            # new_pre_ss = self.my_dx.predict(xu)
            # predict cost
            # xu = np.concatenate((pre_s, np.array([action])))
            xu = np.concatenate((pre_ss.squeeze(), action_s), 1)
            new_pre_ss = self.my_dx.predict(xu)
            pre_r = self.get_actual_cost(pre_ss, action_s)
            # pre_r = self.cost.predict(torch.Tensor(xu).to(device))

            pre_ss = new_pre_ss

            pre_cum_hori_rewards += pre_r.reshape(-1, 1)

            # print('all pre rewards', pre_cum_hori_rewards)
        elite_indices = list(
            map(pre_cum_hori_rewards.tolist().index, heapq.nlargest(self.num_elites, pre_cum_hori_rewards.tolist())))
        best_indice = pre_cum_hori_rewards.tolist().index(max(pre_cum_hori_rewards.tolist()))

        # print(heapq.nlargest(self.num_elites, cum_hori_rewards.tolist()))
        # print('all rewards', cum_hori_rewards)
        # print('all eps length', eps_length)

        return pre_cum_hori_rewards, elite_indices, best_indice


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Pendulum-v0', metavar='ENV',
                        help='env :[Pendulum-v0, CartPole-v0,CartPole-continuous]')
    parser.add_argument('--use_sample', type=bool, default=False, metavar='NS')
    parser.add_argument('--sample_per_epoch', type=bool, default=False, metavar='NS')
    parser.add_argument('--sample_per_batch', type=bool, default=False, metavar='NS')

    parser.add_argument('--predict_with_bias', type=bool, default = True, metavar='NS',
                        help='predict y with bias')
    parser.add_argument('--input_normalize', type=bool, default = False, metavar='NS',
                        help='input normalization')
    parser.add_argument('--num-iters', type=int, default=100, metavar='NS',
                        help='number of iterating the distribution params')
    parser.add_argument('--num-elites', type=int, default=5, metavar='NS', help='number of choosing best params')
    parser.add_argument('--num-trajs', type=int, default=100, metavar='NS',
                        help='number of sampling from params distribution')
    # 500
    parser.add_argument('--test-episodes', type=int, default=100, metavar='NS',
                        help='episode number of testing the trained cem')
    parser.add_argument('--test', dest='test', action='store_true', help='use test mode or train mode')
    parser.add_argument('--filename', default='cem_pendulum_params.json', metavar='M', help='saved params')
    parser.add_argument('--alpha', type=float, default=0., metavar='T',
                        help='Controls how much of the previous mean and variance is used for the next iteration.')
    parser.add_argument('--plan-hor', type=int, default=20, metavar='NS', help='number of choosing best params')
    parser.add_argument('--max-iters', type=int, default=5, metavar='NS', help='iteration of cem')
    parser.add_argument('--epsilon', type=float, default=0.001, metavar='NS', help='threshold for cem iteration')
    parser.add_argument('--gpu-ids', type=int, default=None, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
    parser.add_argument('--train-episodes', type=int, default=100, metavar='NS',
                        help='episode number of testing the trained cem')
    # nn model direction
    parser.add_argument('--load-cost-model-dir',
                        default='logs/nn-cost-BDQN-models/CartPole-continuous-cost-BDQN-mse-model.dat', metavar='LMD',
                        help='folder to load trained models from')
    parser.add_argument('--load-dx-model-dir',
                        default='logs/nn-dx-BDQN-models/CartPole-continuous-dx-BDQN-mse-model.dat', metavar='LMD',
                        help='folder to load trained models from')

    parser.add_argument('--load-cost-param-dir',
                        default='logs/BDQN-cost-params/CartPole-continuous-cost-BDQN-model.dat', metavar='LMD',
                        help='folder to load cost params from')
    parser.add_argument('--load-dx-param-dir', default='logs/BDQN-dx-params/CartPole-continuous-dx-BDQN-model.dat',
                        metavar='LMD', help='folder to load dx params from')
    # no use
    parser.add_argument('--trueenv', dest='trueenv', action='store_true',
                        help='use true env in collecting trajectories')

    # NN
    parser.add_argument('--hidden-dim-dx', type=int, default=200, metavar='NS')

    parser.add_argument('--hidden-dim-cost', type=int, default=20, metavar='NS')

    parser.add_argument('--training-iter-dx', type=int, default=40, metavar='NS', help='iterations in training nn')
    parser.add_argument('--training-iter-cost', type=int, default=60, metavar='NS', help='iterations in training nn')

    # parser.add_argument('--max-collect-eps', type=int, default=300, metavar='NS',
    # help='number of iterating the distribution params')
    parser.add_argument('--collect-step', type=int, default=200, metavar='NS',
                        help='episode number of testing the trained cem')
    parser.add_argument('--lr', type=float, default=0.001, metavar='T', help='learning rate in training nn.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='NS', help='batch size in training nn')
    # parser.add_argument('--data-type', default='random', metavar='M', help='collect data to train nn [random, cem]')
    parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
    # parser.add_argument('--gp-models-dir', default='gp-models/', metavar='LG', help='folder to save logs')
    parser.add_argument('--optimize', dest='optimize', action='store_true', help='if true, use gp optimization')
    # parser.add_argument('--save-data-dir', default=None, metavar='LG', help='folder to save logs')
    parser.add_argument('--mpc-policy', default='cem', metavar='M', help='collect data to train nn [random, cem]')
    # random action, random shooting

    # neural_bayes param
    parser.add_argument('--a0', type=float, default=6.0, metavar='T', help='a0')
    parser.add_argument('--b0', type=float, default=6.0, metavar='T', help='b0')
    parser.add_argument('--lambda_prior', type=float, default=0.25, metavar='T', help='lambda_prior')
    parser.add_argument('--var_scale', type=float, default=4.0, metavar='T', help='var_scale')

    parser.add_argument('--if_save', type=bool, default=False, metavar='T', help='save model params or not')

    args = parser.parse_args()
    print("current dir:", os.getcwd())
    if 'CartPole-continuous' in args.env:
        env = ContinuousCartPoleEnv()
    elif 'Pendulum-v0' in args.env:
        env = PendulumEnv()
    else:
        env = gym.make(args.env)
    if args.gpu_ids is not None:
        device = torch.device('cuda:' + '2')
    else:
        device = torch.device('cpu')
    print('use device', device)
    print('env', env)
    if 'CartPole-v0' in args.env:
        slb = env.observation_space.low  # state lower bound
        sub = env.observation_space.high  # state upper bound
        alb = np.zeros(1)  # action lower bound
        aub = np.ones(1)  # action upper bound
        # print(slb, sub, alb, aub)
    else:
        slb = env.observation_space.low
        sub = env.observation_space.high
        alb = env.action_space.low
        aub = env.action_space.high
        # print(slb, sub, alb, aub)
    obs_shape = env.observation_space.shape[0]
    action_shape = len([env.action_space.sample()])
    dx_model = construct_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)

    my_dx = neural_bays_dx_tf(args, dx_model, "dx", obs_shape, device, sigma_n=0.01**2,sigma=0.01**2)


    cem = CEM(env, args, my_dx, num_elites=args.num_elites, num_trajs=args.num_trajs, alpha=args.alpha,
              device=device)

    num_episode = 15
    rewards = []
    for episode in range(num_episode):
        state = torch.tensor(env.reset())
        if 'Pendulum-v0' in args.env:
            state = state.squeeze()
        time_step = 0
        done = False
        length = 0
        cum_rewards = 0
        my_dx.sample_BDQN()

        avg_st_loss = 0
        avg_cost_loss = 0

        num_steps = 200
        for _ in range(num_steps):
            best_action = cem.hori_planning(state)
            # print('best_action', best_action)
            if 'CartPole-v0' in args.env:
                best_action = 1 if best_action >= 0 else 0
            elif 'Pendulum-v0' in args.env:
                best_action = np.array([best_action])
            # best_action = env.action_space.sample()
            new_state, r, done, _ = env.step(best_action)
            r += np.random.normal(0,0.01)
            r = torch.tensor(r)
            new_state = torch.tensor(new_state)
            if 'Pendulum-v0' in args.env:
                new_state = new_state.squeeze()
                best_action = best_action.squeeze(0)

            xu = torch.cat((state.double(), torch.tensor(best_action).double()))
            # print('step', time_step, 'reward', r, 'done', done)
            # print('real state',new_state,'predict',predict_state)
            # print('state diff', torch.tensor(new_state).float()-predict_state)
            # print('state l1 loss',eva_loss(torch.tensor(new_state).unsqueeze(0),predict_state))
            # print('real cost', r, 'predict', pre_r)
            # print('cost diff', torch.tensor(r).float()-pre_r)
            # print('cost l1 loss',eva_loss(torch.tensor(r).float(),pre_r.float()))
            # my_cost.add_data(new_x=xu, new_y=r)
            my_dx.add_data(new_x=xu, new_y=new_state - state)

            if episode >= 1:
                predict_state = my_dx.predict(xu.numpy().reshape(1,-1))
                # pre_r = my_cost.predict(xu.float())
                eva_loss = torch.nn.L1Loss()
                avg_st_loss += eva_loss(torch.tensor(new_state).unsqueeze(0), torch.tensor(predict_state))
                # avg_cost_loss += eva_loss(torch.tensor(r).float(),pre_r.float())
            # env.render()
            time_step += 1
            cum_rewards += r
            length += 1
            state = new_state

            # if done:
            # print(episode, ': cumulative rewards', cum_rewards, 'length', length)
        print(episode, ': cumulative rewards', cum_rewards)
        print('avg st loss: ', avg_st_loss / num_steps)
        # print('avg cost loss: ', avg_cost_loss/num_steps)
        rewards.append([episode, cum_rewards.tolist()[0]])

        my_dx.train(epochs = 100)
        my_dx.update_bays_reg_BDQN()
        # my_cost.train()
        # my_cost.update_bays_reg_BDQN()
        # print("my dx cov: {}".format(my_dx.cov_w[:, :3, :3]))
        # print("my cost cov: {}".format(my_cost.cov_w[:, :3, :3]))
    np.savetxt('pen.txt', np.array(rewards))