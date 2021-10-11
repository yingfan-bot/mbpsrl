Please see requirements.txt for package dependencies.

Download folder: MPC-PSRL
cd ./MPC-PSRL

Directly run python files for each environment:

Stochastic Cartpole: 
python CEM_CPCR_tf.py 
(with oracle rewards)

python CEM_CPC_tf.py 
(without oracle rewards)

Stochastic Pendulum: 
python CEM_penr_tf.py 
(with oracle rewards)
python CEM_pen_tf.py 
(without oracle rewards)

Reacher:
python CEM_reacher_tf.py 
(with oracle rewards)
python CEM_reacher_without_tf.py 
(without oracle rewards)

Pusher: 
python CEM_pusher_tf.py 
(with oracle rewards)
python CEM_pusher_without_tf.py 
(without oracle rewards)

Cumulative rewards are saved in folder MPC-PSRL as .txt files.
