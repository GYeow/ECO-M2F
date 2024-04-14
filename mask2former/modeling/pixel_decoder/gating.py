import torch
from torch import nn
import torch.nn.functional as F

class GatingFunction(nn.Module):
    
    def __init__(self,
                d_model=256,
                num_gates_cls=5):
        super().__init__()
        
        self.avgpool_in = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, num_gates_cls)

        nn.init.constant_(self.head.bias, 0)


    def forward(self, x):
        # x =  (B, N, C); N=512 for cityscapes
        x = self.avgpool_in(x.permute(0, 2, 1)) # (B, N, C) -> (B, C, 1)

        x = torch.flatten(x, 1) # B C

        x = self.head(x) # B num_gates_cls

        return x