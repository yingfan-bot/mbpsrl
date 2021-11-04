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
from CEM_with import CEM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    parser.add_argument('--max-iters', type=int, default=5, metavar='NS', help='iteration of cem')

    parser.add_argument('--train-episodes', type=int, default=100, metavar='NS', help='episode number of testing the trained cem')

    parser.add_argument('--var', type=float, default=10.0, metavar='T', help='var')



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

    dx_model = construct_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)
    # cost_model = construct_cost_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)


    my_dx = neural_bays_dx_tf(args, dx_model, "dx", obs_shape, sigma_n2=1e-4**2,sigma2=1e-3**2)
    # my_cost = neural_bays_dx_tf(args, cost_model, "cost", 1, sigma_n2=1e-4**2,sigma2=1e-3**2)






    avg_loss = []
    num_episode = 30
    for episode in range(num_episode):
        cem = CEM(env, args, my_dx, num_elites=args.num_elites, num_trajs=args.num_trajs, alpha=args.alpha)

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
        my_dx.sample()
        # my_cost.sample()
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
            # my_cost.add_data(new_x=xu, new_y= r)

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

        my_dx.train(epochs = 50)
        my_dx.update_bays_reg()
        # if args.input_normalize:
        # #     my_dx.fit_input_stats()
        #     my_cost.fit_input_stats()
        # my_cost.train(epochs = 50)
        # my_cost.update_bays_reg()

        print(avg_loss)
        # save_dir = os.path.join("./logs/", "grid_search_id{}.txt".format(args.id))
        # with open(save_dir, "w") as f:
        #     json.dump(avg_loss, f)
        # np.savetxt('without_reacher.txt',np.array(avg_loss))