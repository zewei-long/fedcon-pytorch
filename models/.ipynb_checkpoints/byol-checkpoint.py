import copy
import random 
import torch 
from torch import nn 
import torch.nn.functional as F 
from torchvision import transforms 
from math import pi, cos 
from collections import OrderedDict

HPS = dict(
    max_steps=int(1000. * 1281167 / 4096), # 1000 epochs * 1281167 samples / batch size = 100 epochs * N of step/epoch
    # = total_epochs * len(dataloader) 
    mlp_hidden_size=4096*3,
    projection_size=4096,
    base_target_ema_g=4e-4,#4e-4
    base_target_ema_l=4e-4,
    optimizer_config=dict(
        optimizer_name='lars', 
        beta=0.9, 
        trust_coef=1e-3, 
        weight_decay=1.5e-6,
        exclude_bias_from_adaption=True),
    learning_rate_schedule=dict(
        base_learning_rate=0.2,
        warmup_steps=int(10.0 * 1281167 / 4096), # 10 epochs * N of steps/epoch = 10 epochs * len(dataloader)
        anneal_schedule='cosine'),
    batchnorm_kwargs=dict(
        decay_rate=0.9,
        eps=1e-5), 
    seed=1337,
)


# def loss_fn(x, y, version='simplified'):
    
#     if version == 'original':
#         y = y.detach()
#         x = F.normalize(x, dim=-1, p=2)
#         y = F.normalize(y, dim=-1, p=2)
#         return (2 - 2 * (x * y).sum(dim=-1)).mean()
#     elif version == 'simplified':
#         return (2 - 2 * F.cosine_similarity(x,y.detach(), dim=-1)).mean()
#     else:
#         raise NotImplementedError

from .simsiam import D  # a bit different but it's essentially the same thing: neg cosine sim & stop gradient


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, HPS['mlp_hidden_size']),
            nn.BatchNorm1d(HPS['mlp_hidden_size'], eps=HPS['batchnorm_kwargs']['eps'], momentum=1-HPS['batchnorm_kwargs']['decay_rate']),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(HPS['mlp_hidden_size'], in_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class fc_Mnist(nn.Module):
    def __init__(self, in_dim):
        super(fc_Mnist, self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class fc_Cifar(nn.Module):
    def __init__(self, in_dim):
        super(fc_Cifar, self).__init__()

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.fc_layer(x)
        return x

class global_net(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        if backbone.output_dim == 320:
            self.fc = fc_Mnist(backbone.output_dim)
        else:
            self.fc = fc_Cifar(backbone.output_dim)
        
        self.teacher = nn.Sequential(
            self.backbone,
            self.fc
        )

        self.student = copy.deepcopy(self.teacher)


    def target_ema(self, k, K, base_ema=HPS['base_target_ema_g']):
        return 1 - base_ema * (cos(pi*k/K)+1)/2 

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        for online, target in zip(self.teacher.parameters(), self.student.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
            
    def forward(self, x1, x2):
        t, s = self.teacher, self.student

        z1_t = t(x1)
        z2_t = t(x2) 


        with torch.no_grad():
            z1_s = t(x1)
            z2_s = t(x2) 

        return z1_t, z2_t, z1_s, z2_s



class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projector = MLP(backbone.output_dim)

        self.online_encoder = nn.Sequential(
            self.backbone
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)


    def target_ema(self, k, K, base_ema=HPS['base_target_ema_l']):
        return 1 - base_ema * (cos(pi*k/K)+1)/2 

    @torch.no_grad()
    def update_moving_average(self):
        tau = 0.999#self.target_ema(global_step, max_steps)
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
            
    def forward(self, x1, x2):
        f_o, h_o = self.online_encoder, self.projector
        f_t      = self.target_encoder

        z1_o = f_o(x1)
        z2_o = f_o(x2)

        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)

        with torch.no_grad():
            z1_t = f_t(x1)
            z2_t = f_t(x2)
        
        L = D(p1_o, z2_t) / 2 + D(p2_o, z1_t) / 2 

        return L



if __name__ == "__main__":
    pass