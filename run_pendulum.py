import numpy as np
import gym
import heapq
import argparse
import json
import os
from pendulum_gym import PendulumEnv
from cartpole_continuous import ContinuousCartPoleEnv
import torch
from CEM_without import CEM
import scipy.stats as stats
from NB_dx_tf import neural_bays_dx_tf

from tf_models.constructor import construct_shallow_model, construct_shallow_cost_model, construct_model, construct_cost_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Pendulum-v0', metavar='ENV',
                        help='env :[Pendulum-v0, CartPole-v0,CartPole-continuous]')

    parser.add_argument('--predict_with_bias', type=bool, default = True, metavar='NS',
                        help='predict y with bias')
    parser.add_argument('--num-elites', type=int, default=5, metavar='NS', help='number of choosing best params')
    parser.add_argument('--num-trajs', type=int, default=100, metavar='NS',
                        help='number of sampling from params distribution')

    parser.add_argument('--test-episodes', type=int, default=100, metavar='NS',
                        help='episode number of testing the trained cem')
    parser.add_argument('--test', dest='test', action='store_true', help='use test mode or train mode')
    parser.add_argument('--filename', default='cem_pendulum_params.json', metavar='M', help='saved params')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='T',
                        help='Controls how much of the previous mean and variance is used for the next iteration.')
    parser.add_argument('--plan-hor', type=int, default=30, metavar='NS', help='number of choosing best params')
    parser.add_argument('--max-iters', type=int, default=5, metavar='NS', help='iteration of cem')
    parser.add_argument('--epsilon', type=float, default=0.001, metavar='NS', help='threshold for cem iteration')
    parser.add_argument('--gpu-ids', type=int, default=None, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
    parser.add_argument('--train-episodes', type=int, default=100, metavar='NS',
                        help='episode number of testing the trained cem')

    parser.add_argument('--hidden-dim-dx', type=int, default=200, metavar='NS')

    parser.add_argument('--hidden-dim-cost', type=int, default=200, metavar='NS')

    parser.add_argument('--training-iter-dx', type=int, default=40, metavar='NS')
    parser.add_argument('--training-iter-cost', type=int, default=60, metavar='NS')
    parser.add_argument('--var', type=float, default=1.0, metavar='T', help='var')

    args = parser.parse_args()
    print("current dir:", os.getcwd())
    if 'CartPole-continuous' in args.env:
        env = ContinuousCartPoleEnv()
    elif 'Pendulum-v0' in args.env:
        env = PendulumEnv()
    else:
        env = gym.make(args.env)

    print('env', env)

    slb = env.observation_space.low
    sub = env.observation_space.high
    alb = env.action_space.low
    aub = env.action_space.high

    obs_shape = env.observation_space.shape[0]
    action_shape = len([env.action_space.sample()])

    dx_model = construct_shallow_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)
    cost_model = construct_shallow_cost_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=10, num_networks=1, num_elites=1)

    my_dx = neural_bays_dx_tf(args, dx_model, "dx", obs_shape, sigma_n2=0.001**2,sigma2= 1**2)

    my_cost = neural_bays_dx_tf(args, cost_model, "cost", 1, sigma_n2 = 0.001**2,sigma2 = 1**2)



    num_episode = 15
    rewards = []
    for episode in range(num_episode):
        cem = CEM(env, args, my_dx, my_cost, num_elites=args.num_elites, num_trajs=args.num_trajs, alpha=args.alpha)
        state = torch.tensor(env.reset())

        if 'Pendulum-v0' in args.env:
            state = state.squeeze()
        time_step = 0
        done = False
        length = 0
        cum_rewards = 0
        my_dx.sample()
        my_cost.sample()
        avg_st_loss = 0
        avg_cost_loss = 0

        num_steps = 200


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
            my_cost.add_data(new_x=xu, new_y=r)
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
        my_dx.update_bays_reg()
        my_cost.train(epochs = 100)
        my_cost.update_bays_reg()
        print(rewards)

        np.savetxt('pen_without.txt', np.array(rewards))


# for _ in range(5):
#     main()