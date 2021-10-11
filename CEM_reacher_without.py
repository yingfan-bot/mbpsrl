import numpy as np
import gym
import heapq
import argparse
import json
import os
from pendulum_gym import PendulumEnv
from cartpole_continuous import ContinuousCartPoleEnv
from pusher import PusherEnv
from reacher import Reacher3DEnv
import torch
import scipy.stats as stats
from NB_dx_tf import neural_bays_dx_tf
from tf_models.constructor import construct_model, construct_cost_model

import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class CEM():
    def __init__(self, env, args, my_dx,  my_cost, num_elites, num_trajs, alpha, device):
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
        else:
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

        if 'CartPole-v0' not in self.env_name:
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(means), scale=np.ones_like(means))
            lb_dist, ub_dist = means - self.lb, self.ub - means
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
            # constrained_var = np.sqrt(constrained_var)
            samples = X.rvs(size=[self.num_trajs,self.soln_dim]) * np.sqrt(constrained_var) + means
            solution = samples.copy().T
            # solution = np.array([np.clip(np.random.normal(means[t], np.sqrt(vars[t]), self.num_trajs), self.lb, self.ub) for t in range(self.plan_hor*self.action_shape)])
        else:  # discrete action space
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(means), scale=np.ones_like(means))
            lb_dist, ub_dist = means - self.lb, self.ub - means
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
            samples = X.rvs(size=[self.num_trajs,self.soln_dim]) * np.sqrt(constrained_var) + means
            solutions = samples.copy().T
            # solution = np.array([np.random.normal(means[t], np.sqrt(vars[t]), self.num_trajs) for t in range(self.plan_hor*self.action_shape)])
            # lb_dist, ub_dist = means - self.lb, self.ub - means
            # constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
            # constrained_var = np.sqrt(constrained_var)
            # samples = X.rvs(size=[self.num_trajs,self.soln_dim]) * np.sqrt(constrained_var) + means
            # solution = samples.copy().T
            # solution = np.array([np.clip(np.random.normal(means[t], constrained_var[t], self.num_trajs), self.lb, self.ub) for t in range(self.plan_hor*self.action_shape)])

        return solution, means, vars

    # def hori_planning(self, cur_s):
    def hori_planning(self, cur_s):
        cur_s = cur_s.squeeze()
        '''choose elite actions from simulation trajectorys from current timestep t'''
        action_shape = len([self.env.action_space.sample()])
        # init_means = [np.zeros(action_shape) for t in range(plan_hor)]  # for multiple dimenstions of action
        # init_vars = [np.eye(action_shape) for t in range(plan_hor)]
        # init_means = np.zeros(self.action_shape * self.plan_hor)
        # update means
        init_means = np.concatenate((self.pre_means[self.action_shape:],np.zeros(self.action_shape)))

        init_vars = args.var*np.ones(self.action_shape * self.plan_hor)
        means = init_means
        vars = init_vars

        '''first sampling from initial distribution'''
        if 'CartPole-v0' not in self.env_name:
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(means), scale=np.ones_like(means))
            lb_dist, ub_dist = means - self.lb, self.ub - means
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
            samples = X.rvs(size=[self.num_trajs,self.soln_dim]) * np.sqrt(constrained_var) + means
            init_solutions = samples.copy().T
            # init_solutions = np.array([np.clip(np.random.normal(means[t], np.sqrt(vars[t]), self.num_trajs), self.lb, self.ub) for t in range(self.plan_hor*self.action_shape)])
            # lb_dist, ub_dist = means - self.lb, self.ub - means
            # constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
            # constrained_var = np.sqrt(constrained_var)
            # samples = X.rvs(size=[self.num_trajs,self.soln_dim]) * np.sqrt(constrained_var) + means
            # solution = samples.copy().T
            # init_solutions = np.array([np.clip(np.random.normal(means[t], constrained_var[t], self.num_trajs), self.lb, self.ub) for t in range(self.plan_hor*self.action_shape)])
        else:
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(means), scale=np.ones_like(means))
            lb_dist, ub_dist = means - self.lb, self.ub - means
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
            samples = X.rvs(size=[self.num_trajs,self.soln_dim]) * np.sqrt(constrained_var) + means
            init_solutions = samples.copy().T
            # init_solutions = np.array([np.random.normal(means[t], np.sqrt(vars[t]), self.num_trajs) for t in range(self.plan_hor*self.action_shape)])
            # lb_dist, ub_dist = means - self.lb, self.ub - means
            # constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), vars)
            # constrained_var = np.sqrt(constrained_var)
            # samples = X.rvs(size=[self.num_trajs,self.soln_dim]) * np.sqrt(constrained_var) + means
            # solution = samples.copy().T
            # init_solutions = np.array([np.clip(np.random.normal(means[t], constrained_var[t], self.num_trajs), self.lb, self.ub) for t in range(self.plan_hor*self.action_shape)])
        solutions = init_solutions
        iter = 0

        while iter < self.max_iters and np.max(vars) > self.epsilon:

            pre_rewards, elite_indices, best_indice = self.get_elites(cur_s, solutions)
            solutions, means, vars = self.sample_hori_actions(means, vars, solutions, elite_indices)
            iter += 1
        # print("final cum rewards", pre_rewards[best_indice])

        # best_action = solutions[:, best_indice][0:self.action_shape]
        # print('cem iters:',iter)
        best_action = means[0:self.action_shape]
        self.pre_means = means
        # print('best_indice', best_indice)
        # print('choose solutions', solutions[:, best_indice].shape)
        # print('best action', best_action)
        return best_action

    def get_actual_cost(self, obs, acs):


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
        # compute total costs for each trajs and select the top ones
        # cum_hori_rewards = np.zeros(self.num_trajs)
        # eps_length = np.zeros(self.num_trajs)
        # print("actions",sample_hori_actions)
        pre_cum_hori_rewards = np.zeros([self.num_trajs,1])
        # for k in range(self.num_trajs):
        pre_s = cur_s.numpy().copy()
        # concat all trajs started with current state
        pre_ss = [pre_s for i in range(self.num_trajs)]
        pre_ss = np.array(pre_ss)
        for t in range(self.plan_hor):
            action_s = sample_hori_actions[t*self.action_shape:(t+1)*self.action_shape].T.copy()

            xu = np.concatenate((pre_ss.squeeze(), action_s),1)
            new_pre_ss = self.my_dx.predict(xu)
            # new_pre_ss = torch.clamp(new_pre_ss, min = self.env.observation_space.low[0], max = self.env.observation_space.high[0])
            pre_r  = self.cost.predict(torch.Tensor(xu).to(device))
            # pre_r = self.cost.predict(torch.Tensor(xu).to(device))

            pre_ss = new_pre_ss
            # pre_r = pre_r.detach().cpu().numpy()
            pre_cum_hori_rewards += pre_r.reshape(-1, 1)


        # print('all pre rewards', pre_cum_hori_rewards)
        elite_indices = list(map(pre_cum_hori_rewards.tolist().index, heapq.nlargest(self.num_elites, pre_cum_hori_rewards.tolist())))
        best_indice = pre_cum_hori_rewards.tolist().index(max(pre_cum_hori_rewards.tolist()))
        # print("best pre rewards", pre_cum_hori_rewards[best_indice])
        # print(heapq.nlargest(self.num_elites, cum_hori_rewards.tolist()))
        # print('all rewards', cum_hori_rewards)
        # print('all eps length', eps_length)

        return pre_cum_hori_rewards, elite_indices, best_indice

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--id', default='1', metavar='ENV', help='id')

    parser.add_argument('--use_sample', type=bool, default=False, metavar='NS')
    parser.add_argument('--sample_per_epoch', type=bool, default=False, metavar='NS')
    parser.add_argument('--sample_per_batch', type=bool, default=True, metavar='NS')

    parser.add_argument('--predict_with_bias', type=bool, default = True, metavar='NS',
                        help='predict y with bias')
    parser.add_argument('--input_normalize', type=bool, default = False, metavar='NS',
                        help='input normalization')
    # 0.001 0.01
    parser.add_argument('--sigma', type=float, default=1e-04, metavar='T', help='var for betas')
    parser.add_argument('--sigma_n', type=float, default=1e-06, metavar='T', help='var for noise')
    # NNenv <PusherEnv instance>
    parser.add_argument('--hidden-dim-dx', type=int, default = 200, metavar='NS')
    parser.add_argument('--training-iter-dx', type=int, default=50, metavar='NS')

    parser.add_argument('--num-trajs', type=int, default=400, metavar='NS',
                        help='number of sampling from params distribution')
    parser.add_argument('--num-elites', type=int, default=40, metavar='NS', help='number of choosing best params')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='T',
                        help='Controls how much of the previous mean and variance is used for the next iteration.')
    parser.add_argument('--env', default='Reacher', metavar='ENV', help='env :[Pendulum-v0, CartPole-v0,CartPole-continuous]')
    parser.add_argument('--num-iters', type=int, default=100, metavar='NS', help='number of iterating the distribution params')
    parser.add_argument('--plan-hor', type=int, default=25, metavar='NS', help='number of choosing best params')

    # 500
    parser.add_argument('--test-episodes', type=int, default=100, metavar='NS',
                        help='episode number of testing the trained cem')
    parser.add_argument('--test', dest='test', action='store_true', help='use test mode or train mode')
    parser.add_argument('--filename', default='cem_pendulum_params.json', metavar='M', help='saved params')


    parser.add_argument('--max-iters', type=int, default=5, metavar='NS', help='iteration of cem')
    parser.add_argument('--epsilon', type=float, default=0.001, metavar='NS', help='threshold for cem iteration')
    parser.add_argument('--gpu-ids', type=int, default=None, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
    parser.add_argument('--train-episodes', type=int, default=100, metavar='NS', help='episode number of testing the trained cem')
    # nn model direction
    parser.add_argument('--load-cost-model-dir', default='logs/nn-cost-BDQN-models/CartPole-continuous-cost-BDQN-mse-model.dat', metavar='LMD', help='folder to load trained models from')
    parser.add_argument('--load-dx-model-dir', default='logs/nn-dx-BDQN-models/CartPole-continuous-dx-BDQN-mse-model.dat', metavar='LMD', help='folder to load trained models from')

    parser.add_argument('--load-cost-param-dir', default='logs/BDQN-cost-params/CartPole-continuous-cost-BDQN-model.dat', metavar='LMD', help='folder to load cost params from')
    parser.add_argument('--load-dx-param-dir', default='logs/BDQN-dx-params/CartPole-continuous-dx-BDQN-model.dat', metavar='LMD', help='folder to load dx params from')
    # no use
    parser.add_argument('--trueenv', dest='trueenv', action='store_true', help='use true env in collecting trajectories')

    #parser.add_argument('--max-collect-eps', type=int, default=300, metavar='NS',
    #help='number of iterating the distribution params')
    parser.add_argument('--collect-step', type=int, default=200, metavar='NS',
                        help='episode number of testing the trained cem')
    parser.add_argument('--lr', type=float, default=0.001, metavar='T', help='learning rate in training nn.')
    parser.add_argument('--var', type=float, default=10.0, metavar='T', help='var')
    parser.add_argument('--batch-size', type=int, default=128, metavar='NS', help='batch size in training nn')
    # parser.add_argument('--data-type', default='random', metavar='M', help='collect data to train nn [random, cem]')
    parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
    # parser.add_argument('--gp-models-dir', default='gp-models/', metavar='LG', help='folder to save logs')
    parser.add_argument('--optimize', dest='optimize', action='store_true', help='if true, use gp optimization')
    # parser.add_argument('--save-data-dir', default=None, metavar='LG', help='folder to save logs')
    parser.add_argument('--mpc-policy', default='cem', metavar='M', help='collect data to train nn [random, cem]')
    # random action, random shooting

    #neural_bayes param
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
    elif "Pusher" in args.env:
        env = PusherEnv()
    elif "Reacher" in args.env:
        env = Reacher3DEnv()
    else:
        env = gym.make(args.env)
    if args.gpu_ids is not None:
        device = torch.device('cuda:' + '2')
    else:
        device = torch.device('cpu')
    print('use device', device)
    print('env', env)
    if 'CartPole-v0' in args.env:
        slb = env.observation_space.low #state lower bound
        sub = env.observation_space.high #state upper bound
        alb = np.zeros(1) #action lower bound
        aub = np.ones(1) #action upper bound
        # print(slb, sub, alb, aub)
    else:
        slb = env.observation_space.low
        sub = env.observation_space.high
        alb = env.action_space.low
        aub = env.action_space.high
        # print(slb, sub, alb, aub)
    obs_shape = env.observation_space.shape[0]
    action_shape = len(env.action_space.sample())
    # dx_model = DynNet(args, obs_shape, action_shape)
    #cost_model = CostNet(args, obs_shape, action_shape)
    dx_model = construct_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)
    cost_model = construct_cost_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)

    # if args.load_dx_model_dir is not None:
    #     saved_state = torch.load(
    #         args.load_dx_model_dir,
    #         map_location=lambda storage, loc: storage)
    #     dx_model.load_state_dict(saved_state)


    # if args.load_cost_model_dir is not None:
    #     saved_state = torch.load(args.load_cost_model_dir, map_location=lambda storage, loc: storage)
    #     cost_model.load_state_dict(saved_state)


    # transition function
    #my_dx = GPdx.gp_exact_dx(args, dx_model, slb, sub, alb, aub, device)
    # reward function
    #cost = GPcost.gp_exact_cost(args, cost_model, slb, sub, alb, aub, device)
    # load from saved network: warm start
    # transition function
    my_dx = neural_bays_dx_tf(args, dx_model, "dx", obs_shape, device, sigma_n2=1e-4**2,sigma2=1e-3**2)
    my_cost = neural_bays_dx_tf(args, cost_model, "cost", 1, device, sigma_n2=1e-4**2,sigma2=1e-3**2)


    # if  args.load_dx_param_dir is not None:
    #     params_dir =  pickle.load(open(args.load_dx_param_dir, 'rb'))
    #     my_dx.cov_w = params_dir['cov']
    #     my_dx.mu_w =s params_dir['mu']

    # reward function
    #my_cost = neural_bays_dx(args, cost_model, "cost", 1, device, sigma = args.sigma, sigma_n = args.sigma_n)

    # if args.load_cost_param_dir is not None:
    #     params_dir = pickle.load(open(args.load_cost_param_dir, 'rb'))
    #     my_cost.cov_w = params_dir['cov']
    #     my_cost.mu_w = params_dir['mu']




    avg_loss = []
    num_episode = 30
    for episode in range(num_episode):
        cem = CEM(env, args, my_dx, my_cost, num_elites = args.num_elites, num_trajs = args.num_trajs, alpha = args.alpha, device = device)
        # if episode > 19:
        #     cem = CEM(env, args, my_dx, num_elites = args.num_elites, num_trajs = args.num_trajs, alpha = args.alpha, device = device, use_mean = True)
        state = torch.tensor(env.reset())
        if 'Pendulum-v0' in args.env:
            state = state.squeeze()
        time_step = 0
        done = False
        # init_mean = np.zeros(action_shape*args.plan_hor)
        # init_var = 4*np.eye(action_shape*args.plan_hor)
        # mean, var = init_mean, init_var
        # TODO: multidimensional
        # sample_actions = np.random.multivariate_normal(init_mean, init_var, args.num_trajs)
        length = 0
        cum_rewards = 0
        my_dx.sample_BDQN()
        my_cost.sample_BDQN()
        avg_dx_loss = 0
        avg_cost_loss = 0

        num_steps = 150
        for _ in range(num_steps):
            if episode == 0:
                best_action = env.action_space.sample()
            else:
                best_action = cem.hori_planning(state)
            # print('best_action', best_action)
            if 'CartPole-v0' in args.env:
                best_action = 1 if best_action >= 0 else 0
            elif 'Pendulum-v0' in args.env:
                best_action = np.array([best_action])
            # best_action = env.action_space.sample()
            new_state, r, done, _ = env.step(best_action)
            # print("reward", r)
            r = torch.tensor(r)
            new_state = torch.tensor(new_state)
            if 'Pendulum-v0' in args.env:
                new_state = new_state.squeeze()
                best_action = best_action.squeeze(0)

            xu = torch.cat((state.double(),torch.tensor(best_action).double()))
            my_cost.add_data(new_x=xu, new_y= r)

            my_dx.add_data(new_x=xu, new_y=new_state - state)

            if episode >= 1:
                predict_state = my_dx.predict(xu.numpy().reshape(1,-1))
                # pre_r = my_cost.predict(xu.float())
                eva_loss = torch.nn.L1Loss()
                avg_dx_loss += eva_loss(torch.tensor(new_state).unsqueeze(0),torch.tensor(predict_state)).tolist()
                # avg_cost_loss += eva_loss(torch.tensor(r).float(),pre_r.float())
            # env.render()
            time_step += 1
            cum_rewards += r
            length += 1
            state = new_state


            # if done:
            # print(episode, ': cumulative rewards', cum_rewards, 'length', length)
        print(episode, ': cumulative rewards', cum_rewards)
        print('avg dx loss: ', avg_dx_loss/num_steps)
        avg_loss.append([episode, cum_rewards.tolist(), (avg_dx_loss/num_steps)])

        print('avg cost loss: ', avg_cost_loss/num_steps)

        my_dx.train(epochs = 100)
        my_dx.update_bays_reg_BDQN()
        # if args.input_normalize:
        # #     my_dx.fit_input_stats()
        #     my_cost.fit_input_stats()
        my_cost.train(epochs = 100)
        my_cost.update_bays_reg_BDQN()

        print(avg_loss)
        # save_dir = os.path.join("./logs/", "grid_search_id{}.txt".format(args.id))
        # with open(save_dir, "w") as f:
        #     json.dump(avg_loss, f)
        np.savetxt('without_reacher.txt',np.array(avg_loss))