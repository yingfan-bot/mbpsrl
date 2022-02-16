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

If you find the code useful, please cite:
```
@InProceedings{pmlr-v139-fan21b,
  title = 	 {Model-based Reinforcement Learning for Continuous Control with Posterior Sampling},
  author =       {Fan, Ying and Ming, Yifei},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {3078--3087},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/fan21b/fan21b.pdf},
  url = 	 {https://proceedings.mlr.press/v139/fan21b.html}
```
