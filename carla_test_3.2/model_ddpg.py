import torch
import torch.nn as nn
import torch.nn.functional as F
##from distributions import Categorical, DiagGaussian
from utils import init, init_normc_
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class Policy(nn.Module):
    def __init__(self, obs_space, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        self.img_num_features = obs_space.spaces['img'].shape[0]
      #  print("img_num_features=",self.img_num_features)
        self.v_num_features = obs_space.spaces['v'].shape[0]
     #   print("v_num_features=",self.v_num_features)
        self.num_outputs = action_space.shape[0]
        self.base = CNNBase(self.img_num_features, self.v_num_features,self.num_outputs)
        #print(self.base)

    def forward(self, img, v):
        raise NotImplementedError

    def action(self, img, v):
        _,a=self.base(img,v)
        return a


    def get_value(self, img, v):
        q_value, _= self.base(img, v)
        return q_value


class CNNBase(nn.Module):
    def __init__(self, img_num_features, v_num_features,num_outputs,init_w=3e-4, hidden_size=512):
        super(CNNBase, self).__init__()
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
     #  print(self.cnn_backbone)
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
        self.qfc_joint=nn.Sequential(
            init_(nn.Linear(hidden_size*2+2,hidden_size*2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size*2, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU()
        )
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.fc_means=nn.Linear( hidden_size,num_outputs)
        self.fc_means.bias.data[0] = 0.1 # Throttle
        self.fc_means.bias.data[1] = 0.0 # steer
        self.fc_means.weight.data.fill_(0)
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()
##a 代表action'
    def forward(self, img, v):
        x_img = self.cnn_backbone(img)
        x_v = self.fc_backbone(v)
        a = torch.cat([x_img, x_v], -1)
        import pdb
        breakpoint()
        a = self.fc_joint(a)
        a=self.fc_means(a)   
        q=torch.cat([x_img,x_v,a],-1)
       
        q=self.qfc_joint(q)
        q=self.critic_linear(q)

        
        
        return q, a