##model train exchange
import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import copy
import os
import time
import yaml
import shutil
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
import numpy.random as rd
from carla_logger import setup_carla_logger

import traceback
import argparse
from utils import save_modules, load_modules
import subprocess
import datetime
from tensorboardX import SummaryWriter

# import agents
from arguments import get_args
from util import dict_to_obs, obs_to_dict
from carla_ddpg import DDPGCarla
from  env_test_beta import CarEnv
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
        assert os.path.isfile(args.resume_training), 'Checkpoint file does not exist'
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
    experiment_name = args.experiment_name + '_' + now.strftime("%Y-%m-%d_%H-%M-%S")

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
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard', experiment_name))

    # Logger that writes to STDOUT and a file in the save_dir
    logger = setup_carla_logger(args.save_dir, experiment_name)

    device = torch.device("cuda:1" if args.cuda else "cpu")
    

 

    envs=CarEnv()     
    global agent     
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
    
    for i_episode in range(config.num_updates):
        
        # Save the first observation
        # calc static cycles
        static_cycles = 0
        explore_before_train(envs, agent)
        agent.update(16)  
        obs = envs.reset()   
        obs = obs_to_dict(obs)        
        episode_reward=0

        for episode_timesteps in range(config.num_steps):
            # Sample actions           
            action = agent.get_noise_action(obs, config.act_noise)
     
            if  episode_timesteps == config.num_steps :
                envs.done=True

            # judege speed
            if (envs.obs.get_forward_speed(envs.vehicle)*3.6)<= 0.15:
                static_cycles += 1
            else :
                static_cycles = 0

            if static_cycles > 20:
                print("vechile stay static over 20 frames, break")                
                envs.timeout=True
            # Observe reward and next obs
            next_obs, reward, done, info = envs.step(action)           
            agent.replay_buffer.push(obs,action,reward,next_obs,done) 
            #time.sleep(0.5)
           # import pdb
            #breakpoint()
            agent.update(1)                         
            obs=next_obs         
            episode_reward += reward 
            print(f"No.{episode_timesteps+1},reward={reward}")
           
            if done:
                obs = envs.reset() 
                break
                          
               
        print (f"episode_num={i_episode+1},total_reward={episode_reward},total_num={episode_timesteps+1},ava_reward={episode_reward/(episode_timesteps+1)}")
        writer.add_scalar('carla_avarge_reward',episode_reward/(episode_timesteps+1),i_episode+1)
        writer.add_scalar('episode_num',episode_timesteps+1,i_episode+1)
        writer.add_scalar('carla_total_reward',episode_reward,i_episode+1)
       # agent.update()
        #writer.add_scalar('carla_speed',envs.obs.get_forward_speed(envs.vehicle),i_episode)
             	 
  

        if (args.eval_interval is not None and j % args.eval_interval == 0):
            eval_envs = CarEnv()    

            eval_episode_rewards = []
            obs = eval_envs.reset()
     
            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    action = agent.get_action(
                        obs,config.act_noise)

                # Obser reward and next obs
                carla_obs, reward, done, infos = eval_envs.step(action)

               
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])


            logger.info(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))
    save_path = os.path.join(save_dir_model, str(1) + '.pth.tar')
    save_modules(agent.optimizer, agent.actor, args, config, save_path)
    save_modules(agent.optimizer, agent.critic, args, config, save_path)

def explore_before_train(envs, agent):
    # just for off-policy. Because on-policy don't explore before training.
    print("exploration  start")
    obs = envs.reset()
    obs = obs_to_dict(obs)
    for step in range(300):       
        action=[]
        throttle = rd.uniform(-0.001, 0.4)
        steer = rd.normal(0,1)
       # steer=rd.uniform(-0.3,0.3)
        action.append(throttle)
        action.append(steer)
        action=np.array(action)
        action=np.clip(action,-0.3,0.3)        
       # import pdb
       # breakpoint()
        next_obs, reward, done, info = envs.step(action)
        time.sleep(0.3)       
        agent.replay_buffer.push(obs,action,reward,next_obs,done) 
        obs = next_obs
        print(f"No.{step+1},reward={reward}")
        if done:
            print("exploration is over")
            envs.reset()
            break


if __name__ == "__main__":
    main()
