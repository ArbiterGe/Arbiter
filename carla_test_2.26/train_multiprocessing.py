# model train exchange
import glob
import os
import sys
import threading
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import argparse
import copy
import datetime
import os
import shutil
import subprocess
import time
import traceback
from collections import namedtuple

import carla
import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from IPython import embed
from tensorboardX import SummaryWriter

# import agents
from arguments import get_args
from carla_ddpg import DDPGCarla
from carla_logger import setup_carla_logger
from env_test_beta import CarEnv
from util import dict_to_obs, obs_to_dict
from utils import load_modules, save_modules

FLAG=True

#import subprocess
#from pathlib import Path


def get_config_and_checkpoint(args):
    config_dict, checkpoint = None, None
    if args.config and args.resume_training:
        print('ERROR: Should either provide --config or --resume-training but not both.')
        exit(1)

    if args.config:
        config_dict = load_config_file(args.config)

    if args.resume_training:
        print('Resuming training from: {}'.format(args.resume_training))
        assert os.path.isfile(
            args.resume_training), 'Checkpoint file does not exist'
        checkpoint = torch.load(args.resume_training)
        config_dict = checkpoint['config']

    if config_dict is None:
        print("ERROR: --config or --resume-training flag is required.")
        exit(1)

    config = namedtuple('Config', config_dict.keys())(*config_dict.values())
    return config, checkpoint


def load_config_file(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

        # To be careful with values like 7e-5
        config['lr'] = float(config['lr'])
        config['eps'] = float(config['eps'])
        config['alpha'] = float(config['alpha'])
        return config


def set_random_seeds(args, config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if args.cuda:
        torch.cuda.manual_seed(config.seed)
    # TODO: Set CARLA seed (or env seed)


def main():
    config = None
    args = get_args()
    config, checkpoint = get_config_and_checkpoint(args)

    set_random_seeds(args, config)
    eval_log_dir = args.save_dir + "_eval"
    try:
        os.makedirs(args.save_dir)
        os.makedirs(eval_log_dir)
    except OSError:
        pass

    now = datetime.datetime.now()
    experiment_name = args.experiment_name + \
        '_' + now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create checkpoint file
    save_dir_model = os.path.join(args.save_dir, 'model', experiment_name)
    save_dir_config = os.path.join(args.save_dir, 'config', experiment_name)
    try:
        os.makedirs(save_dir_model)
        os.makedirs(save_dir_config)
    except OSError as e:
        logger.error(e)
        exit()

    if args.config:
        shutil.copy2(args.config, save_dir_config)

    # Tensorboard Logging
    writer = SummaryWriter(os.path.join(
        args.save_dir, 'tensorboard', experiment_name))

    # Logger that writes to STDOUT and a file in the save_dir
    logger = setup_carla_logger(args.save_dir, experiment_name)

    device = torch.device("cuda:1" if args.cuda else "cpu")
    envs = CarEnv()
    agent = DDPGCarla(envs.observation_space,
                      envs.action_space,
                      config.batch_size,
                      config.polyak,
                      config.gamma,
                      base_rl=config.base_rl,
                      lr=config.lr,
                      eps=config.eps,
                      replay_size=config.replay_size
                      )

    if checkpoint is not None:
        load_modules(agent.optimizer, agent.model, checkpoint)
    colloct_data_thread = threading.Thread(
        target=collect_data, name='collet_data', args=(envs, agent, config, writer))
    agent_update_thread = threading.Thread(
        target=update_agent, name='agent_update', args=(agent, config,save_dir_model,args))
    colloct_data_thread.start()
    agent_update_thread.start()
    agent_update_thread.join()   
    colloct_data_thread.join()     
    save_path = os.path.join(save_dir_model, str(1) + '.pth.tar')
    save_modules(agent.optimizer, agent.actor, args, config, save_path)
    save_modules(agent.optimizer, agent.critic, args, config, save_path)

def complete_train(envs,agent):
    print("exploration  start")
    obs = envs.reset()
    obs = obs_to_dict(obs)
    for step in range(300):
        action = []
        throttle = rd.uniform(-0.001, 0.4)
        steer =0
        action.append(throttle)
        action.append(steer)
        action = np.array(action)
        action = np.clip(action, -0.3, 0.3)
        if step==299:
             envs.done=True  
        next_obs, reward, done, info = envs.step(action)
        time.sleep(0.4)
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        print(f"No.{step+1},reward={reward}")
        if done:
            print("exploration is over")
            envs.reset()
            break


def explore_before_train(envs, agent):
    # just for off-policy. Because on-policy don't explore before training.
    print("exploration  start")
    obs = envs.reset()
    obs = obs_to_dict(obs)
    for step in range(300):
        action = []
        throttle = rd.uniform(-0.001, 0.4)
        steer = rd.normal(0, 1)
       # steer=rd.uniform(-0.3,0.3)
        action.append(throttle)
        action.append(steer)
        action = np.array(action)
        action = np.clip(action, -0.3, 0.3)
        if step==299:
             envs.done=True
        next_obs, reward, done, info = envs.step(action)
        time.sleep(0.4)
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        print(f"No.{step+1},reward={reward}")
        if done:
            print("exploration is over")
            envs.reset()
            break


def collect_data(envs, agent, config, writer):
    complete_train(envs,agent)
    for i_episode in range(config.num_updates):
        if len(agent.replay_buffer) > config.batch_size:
            global FLAG
            FLAG=False
        static_cycles = 0
        if (i_episode+1)%10==0:
            complete_train(envs,agent)
        if (i_episode+1)%100==0:
            eval_data(envs,agent,writer)
        explore_before_train(envs, agent)
        obs = envs.reset()
        obs = obs_to_dict(obs)
        episode_reward = 0           
        for episode_timesteps in range(config.num_steps):
            action = agent.get_noise_action(obs, config.act_noise)
            if (episode_timesteps+1) == config.num_steps:
                envs.done = True
            # judege speed
            if (envs.obs.get_forward_speed(envs.vehicle)*3.6) <= 0.15:
                static_cycles += 1
            else:
                static_cycles = 0
            if static_cycles > 20:
                print("vechile stay static over 20 frames, break")
                envs.timeout = True
            # Observe reward and next obs
            next_obs, reward, done, info = envs.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            time.sleep(0.4)
            print(f"No.{episode_timesteps+1},reward={reward}")
            if done:
                obs = envs.reset()
                break
        print(
            f"episode_num={i_episode+1},total_reward={episode_reward},total_num={episode_timesteps+1},ava_reward={episode_reward/(episode_timesteps+1)}")
        writer.add_scalar('carla_avarge_reward', episode_reward /
                          (episode_timesteps+1), i_episode+1)
        writer.add_scalar('episode_num', episode_timesteps+1, i_episode+1)
        writer.add_scalar('carla_total_reward', episode_reward, i_episode+1)

def eval_data(envs,agent,writer): 
    print("evaluation is start")
    obs = envs.reset()
    obs = obs_to_dict(obs)
    for step in range(300):
        action = agent. get_determ_action(obs)
        if step==299:
            envs.done=True
        next_obs, reward, done, info = envs.step(action)
        time.sleep(0.4)
        obs = next_obs
        writer.add_scalar('eval_test', reward, step)
        print(f"No.{step+1},reward={reward}")
        if done:
            print("evaluation is over")
            envs.reset()
            break


def update_agent(agent, config,save_dir_model,args):
     while(FLAG):
        print("wait thread collect date....")
        time.sleep(2)
     for i in range(10000):
               agent.update(16)
               print(f"No.{i+1},update_agent_threading is running")
               if  (i+1)%50==0:
                    save_path = os.path.join(save_dir_model, str(1) + '.pth.tar')
                    save_modules(agent.optimizer, agent.actor, args, config, save_path)
                    save_modules(agent.optimizer, agent.critic, args, config, save_path)

               


if __name__ == "__main__":
    main()
