# Code for paper: Model-based Reinforcement Learning for Continuous Control with Posterior Sampling (https://arxiv.org/abs/2012.09613)
Please see requirements.txt for package dependencies.

## Directly run files with configuration for each environment:

Stochastic Cartpole: 
```
python run_cartpole.py --with-reward True
```
(with oracle rewards)
```
python run_cartpole.py --with-reward False
```
(without oracle rewards)

Stochastic Pendulum: 
```
python run_pendulum.py --with-reward True
```
(with oracle rewards)
```
python run_pendulum.py --with-reward False
```
(without oracle rewards)

Reacher:
```
python run_reacher --with-reward True
```
(with oracle rewards)
```
python run_reacher.py --with-reward False
```
(without oracle rewards)

Pusher: 
```
python run_pusher.py --with-reward True
```
(with oracle rewards)
```
python run_pusher.py --with-reward False
```
(without oracle rewards)

Cumulative rewards are saved as envname_log.txt files.
