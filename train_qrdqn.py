import os
import yaml
import argparse
from datetime import datetime
import gym

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import QRDQNAgent
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
        '/path/to/tensorboard_logs/fqf', args.env_id, f'QRDQN-{args.seed}-{time}-{hash}')

    # Create the agent and run.
    agent = QRDQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'qrdqn_flatgym.yaml'))
    parser.add_argument('--env_id', type=str, default='CartPole-v0')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
