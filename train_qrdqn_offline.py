import os
import yaml
import argparse
from datetime import datetime
import gym
import gym_stochastic.envs.simplebandits

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import QRDQNAgent, QRDQN_Offline_Agent
import random

def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = gym.make(args.env_id)
    test_env = gym.make(args.env_id)


    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    hash = str(random.randint(1000,9999))
    log_dir = os.path.join(
        '/path/to/tensorboard_logs/fqf2', args.env_id, f'QRDQN-{args.seed}-{time}-{hash}')

    # Create the agent and run.
    online_agent = QRDQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    online_agent.run()

    offline_config = config.copy()
    print(offline_config)
    offline_config['start_steps']=1
    offline_config['batch_size']=32
    offline_config['lr']=1e-3
    offline_config['log_interval']= 200
    offline_config['eval_interval'] = 2000

    time = datetime.now().strftime("%Y%m%d-%H%M")
    hash = str(random.randint(1000,9999))
    log_dir = os.path.join(
        '/path/to/tensorboard_logs/fqf2', args.env_id, f'QRDQN_Ofln-{args.seed}-{time}-{hash}')

    offline_agent = QRDQN_Offline_Agent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **offline_config)
    offline_agent.memory = online_agent.memory
    smp = offline_agent.memory.sample(32)
    offline_agent.train_offline(batches=200000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'qrdqn_offline_flatgym.yaml'))
    #parser.add_argument('--env_id', type=str, default='MultiarmedBanditFixed-v0')
    parser.add_argument('--env_id', type=str, default='CartPole-v0')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
