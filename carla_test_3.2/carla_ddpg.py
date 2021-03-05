##exchange model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
#from torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import torch.nn.functional as F
import random
from ddpg_model_ex import Actor,Critic
from copy import deepcopy
from torch.autograd import Variable

class DDPGCarla(object):
     
    def __init__(self ,
                  observation_space,
                  action_space,                 
                  batch_size ,
                  polyak ,
                  gamma,
                  file,
                  base_rl=None,                 
                  lr=None , 
                  eps=None ,
                  replay_size=None ,):
          self.observation_space=observation_space
          self.action_space=action_space
          self.replay_size=replay_size
          self.batch_size=batch_size
          self.polyak=polyak
          self.gamma=gamma
        
          self.critic=Critic(self.observation_space,
                                self.action_space).to("cuda:1")
          self.targ_critic=deepcopy(self.critic)
          self.actor=Actor(self.observation_space,
                                self.action_space).to("cuda:1")
          self.targ_actor=deepcopy(self.actor)
          self.optimizer = optim.Adam([{'params': self.actor.parameters(), 'lr': lr ,'eps': eps},
                                                                      {'params': self.critic.parameters(), 'lr':lr ,'eps':eps}])
          self.rl_scheduler = CyclicLR(self.optimizer,base_lr=base_rl, max_lr=lr,cycle_momentum=False) 
          self.replay_buffer= ReplayBufferH5py(replay_size,file)
          self.act_limit=self.action_space.high[0] 

    # Set up function for computing DDPG Q-loss

    
    def get_noise_action(self, inputs, noise_scale):
        img_tensor=torch.FloatTensor(inputs['img'].astype(float)).to("cuda:1")        
        img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False)
        img_tensor=torch.transpose(img_tensor,1,3)
        v_tensor= torch.FloatTensor(inputs['v'].astype(float)).to("cuda:1")
        v_tensor = Variable(torch.unsqueeze(v_tensor, dim=0).float(), requires_grad=False)
       # print('v_tensor',v_tensor.shape)
        action=self.actor.action(img_tensor,v_tensor)
       # import pdb
      #  breakpoint()
        action=action.detach().cpu().numpy()
        action=action[0]
        action += noise_scale * np.random.randn(self.actor.num_outputs)
        return np.clip(action, -self.act_limit, self.act_limit)

    def get_action(self, inputs):
        img_tensor= []
        v_tensor= []
        for tensor in inputs:
            img_tensor.append(tensor['img']) 
            v_tensor.append(tensor['v'])
        img_tensor=np.array(img_tensor)
        v_tensor=np.array(v_tensor)               
        img_tensor=torch.FloatTensor(img_tensor.astype(float)).to("cuda:1")        
        img_tensor=torch.transpose(img_tensor,1,3)
        v_tensor= torch.FloatTensor(v_tensor.astype(float)).to("cuda:1")     
        action=self.actor.action(img_tensor,v_tensor)
        action=action.detach().cpu().numpy()
       # action=action[0]
        return np.clip(action, -self.act_limit, self.act_limit)
     
    def  get_determ_action(self,inputs):
        img_tensor=torch.FloatTensor(inputs['img'].astype(float)).to("cuda:1")        
        img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False)
        img_tensor=torch.transpose(img_tensor,1,3)
        v_tensor= torch.FloatTensor(inputs['v'].astype(float)).to("cuda:1")
        v_tensor = Variable(torch.unsqueeze(v_tensor, dim=0).float(), requires_grad=False)
        action=self.actor.action(img_tensor,v_tensor)
        action=action.detach().cpu().numpy()
        action=action[0]
        return action
        
    

    def get_targ_value(self,inputs,a):
        img_tensor= []
        v_tensor= []
        for tensor in inputs:
            img_tensor.append(tensor['img'])
            v_tensor.append(tensor['v'])
        img_tensor=np.array(img_tensor)
        v_tensor=np.array(v_tensor) 
        img_tensor=torch.FloatTensor(img_tensor.astype(float)).to("cuda:1")        
        img_tensor=torch.transpose(img_tensor,1,3)
        v_tensor= torch.FloatTensor(v_tensor.astype(float)).to("cuda:1")
        action_tensor= torch.FloatTensor(a.astype(float)).to("cuda:1")    

        return self.targ_critic.get_value(img_tensor,v_tensor,action_tensor)
    def get_value(self, inputs,a):
        img_tensor= []
        v_tensor= []
        for tensor in inputs:
            img_tensor.append(tensor['img'])
            v_tensor.append(tensor['v'])        
        img_tensor=np.array(img_tensor)
        v_tensor=np.array(v_tensor)               
        img_tensor=torch.FloatTensor(img_tensor.astype(float)).to("cuda:1")        
        img_tensor=torch.transpose(img_tensor,1,3)
        v_tensor= torch.FloatTensor(v_tensor.astype(float)).to("cuda:1")  
        action_tensor= torch.FloatTensor(a.astype(float)).to("cuda:1")    
                
        return self.critic.get_value(img_tensor, v_tensor,action_tensor)

    def compute_loss(self,obs,action,next_obs,reward,done):
        q =self. get_value(obs,action)
        # Bellman backup for Q function
        with torch.no_grad():            
            q_targ = self.get_targ_value(next_obs,action)
            backup = reward + self.gamma * (1 - done) * q_targ
        # MSE loss against Bellman backup
        loss_q = ((q - backup.detach())**2).mean()
        p_action=self.get_action(obs)
        loss_p=self.get_value(obs,p_action).mean()
        loss=loss_p-loss_q
        return loss   
    
    def update(self,num):
        for i in range(num):       
            if len(self.replay_buffer) < self.batch_size:
                    return
            obs, action, reward, next_obs, done = self.replay_buffer.sample(
                     self.batch_size)          
            #action = torch.FloatTensor(action).to("cuda:1")
            reward = torch.FloatTensor(reward).unsqueeze(1).to("cuda:1")
            done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to("cuda:1")       
            self.optimizer.zero_grad()
            loss = self.compute_loss(obs,action,next_obs,reward,done)
            loss.backward()      
            self.optimizer.step()
            self.rl_scheduler.step()  
            with torch.no_grad():
                  for p, p_targ in zip(self.actor.parameters(), self.targ_actor.parameters()):                
                        p_targ.data.mul_(self.polyak)
                        p_targ.data.add_((1 - self.polyak) * p.data)
                  for p, p_targ in zip(self.critic.parameters(), self.targ_critic.parameters()):                
                        p_targ.data.mul_(self.polyak)
                        p_targ.data.add_((1 - self.polyak) * p.data)
                    

    

   
##Buffer
class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.stack, zip(*batch))
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(self.buffer)


class ReplayBufferH5py:    
    def __init__(self, capacity,file):
        self.capacity = capacity
        self.file = file
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        self.file.create_group(str(self.position))
        self.file[str(self.position)].create_group("state")
        self.file[str(self.position)].create_group("action")
        self.file[str(self.position)].create_group("reward")
        self.file[str(self.position)].create_group("next_state")
        self.file[str(self.position)].create_group("done")
        self.file[str(self.position)]["state"].create_dataset("img",data=state["img"])
        self.file[str(self.position)]["state"].create_dataset("v",data=state["v"])
        self.file[str(self.position)]["action"].create_dataset("action",data=action)
        self.file[str(self.position)]["next_state"].create_dataset("img",data=next_state["img"])
        self.file[str(self.position)]["next_state"].create_dataset("v",data=next_state["v"])
        self.file[str(self.position)]["reward"].create_dataset("reward",data=reward)
        self.file[str(self.position)]["done"].create_dataset("done",data=done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(list(self.file.keys()), batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch=[],[],[],[],[]
        #batch=np.random.randint(0,self.capacity,size=batch_size)
        for k in batch:
            state={'img':self.file[str(k)]['state']['img'][()],'v':self.file[str(k)]['state']['v'][()]}
            action=self.file[str(k)]['action']['action'][()]
            reward=self.file[str(k)]['reward']['reward'][()]
            next_state={'img':self.file[str(k)]['next_state']['img'][()],'v':self.file[str(k)]['next_state']['v'][()]}
            done=self.file[str(k)]['done']['done'][()]
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(list(self.file.keys()))