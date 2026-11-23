import os
import yaml
import argparse
from datetime import datetime
import gym
from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import IQNAgent, QRDQNAgent, QRDQN_Offline_Agent
import fqf_iqn_qrdqn
import gym

from telepath_worker.ext_replay import PdPlainBuffer, VxBuffer

import pandas as pd


iscuda=True
args_seed=42
args_config="config/qrdqn_offline_flatgym.yaml"
with open(args_config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

env_id="CartPole-v0"
env = gym.make(env_id)
test_env = env

time = datetime.now().strftime("%Y%m%d-%H%M")
log_dir = os.path.join( '/path/to/tensorboard_logs/fqf2', env_id, f'QROf-vx-{args_seed}-{time}')

# Create the agent and run.
agent = QRDQN_Offline_Agent(
env=env, test_env=test_env, log_dir=log_dir, seed=args_seed,
cuda=iscuda, **config)

rloc='/path/to/telepath-runs/cartpole/replay/extended_replay4.csv'
rloc='/path/to/telepath-runs/cartpole/replay/extended_replay4.10000000.csv'
agent.memory = VxBuffer(rloc)
agent.memory.read_from_location()
agent.train_offline(batches=2000000)

agent.save_models('savedir')
