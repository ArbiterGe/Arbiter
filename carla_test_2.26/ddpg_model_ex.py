import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init, init_normc_
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class Critic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        self.img_num_features = obs_space.spaces['img'].shape[0]
        self.v_num_features = obs_space.spaces['v'].shape[0]
        self.num_outputs = action_space.shape[0]
        self.base = Critic_Base(self.img_num_features, self.v_num_features,self.num_outputs)
    def forward(self, img, v,a):
        raise NotImplementedError
    def get_value(self, img, v,a):
        q_value= self.base(img, v,a)
        return q_value

class Critic_Base(nn.Module):
    def __init__(self, img_num_features, v_num_features,num_outputs,init_w=3e-4, hidden_size=512):
        super(Critic_Base, self).__init__()
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))        
        self.cnn_backbone = nn.Sequential(
            init_(nn.Conv2d(img_num_features, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 4, stride=3)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 16, 2, stride=2)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(16 *6 *6, hidden_size)),
            nn.ReLU()
        )
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.fc_backbone = nn.Sequential(
            init_(nn.Linear(v_num_features, hidden_size//4)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size//4, hidden_size//2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size//2, hidden_size)),
            nn.ReLU()
        )
        self.qfc_joint=nn.Sequential(
            init_(nn.Linear(hidden_size*2+2,hidden_size*2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size*2, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()
##a 代表action'
    def forward(self, img, v,a):
        x_img = self.cnn_backbone(img)
        x_v = self.fc_backbone(v)  
      # import pdb
      #  breakpoint()
        q=torch.cat([x_img,x_v,a],-1)       
        q=self.qfc_joint(q)
        q=self.critic_linear(q)         
        return q


class Actor(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.img_num_features = obs_space.spaces['img'].shape[0]
        self.v_num_features = obs_space.spaces['v'].shape[0]
        self.num_outputs = action_space.shape[0]
        self.base = Actor_Base(self.img_num_features, self.v_num_features,self.num_outputs)
    def forward(self, img, v):
        raise NotImplementedError
    def action(self, img, v):
       a=self.base(img,v)
       return a

class Actor_Base(nn.Module):
    def __init__(self, img_num_features, v_num_features,num_outputs,init_w=3e-4, hidden_size=512):
        super(Actor_Base, self).__init__()
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        
        self.cnn_backbone = nn.Sequential(
            init_(nn.Conv2d(img_num_features, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 4, stride=3)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 16, 2, stride=2)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(16 *6 *6, hidden_size)),
            nn.ReLU()
        )
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.fc_backbone = nn.Sequential(
            init_(nn.Linear(v_num_features, hidden_size//4)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size//4, hidden_size//2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size//2, hidden_size)),
            nn.ReLU()
        )
        self.fc_joint = nn.Sequential(
            init_(nn.Linear(hidden_size*2, hidden_size*2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size*2, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )
        self.fc_means=nn.Linear( hidden_size,num_outputs)
        self.fc_means.bias.data[0] = 0.1 # Throttle
        self.fc_means.bias.data[1] = 0.0 # steer
        self.fc_means.weight.data.fill_(0)
        self.train()
##a 代表action'
    def forward(self, img, v):
        x_img = self.cnn_backbone(img)
        x_v = self.fc_backbone(v)
        a = torch.cat([x_img, x_v], -1)
        a = self.fc_joint(a)
        a=self.fc_means(a)        
        return  a