import torch

# 随机深度模块
def drop_path_f(x, drop_prob: float = 0., training: bool=False):
    
    if drop_prob == 0. or not training:
        return x 
    
    keep_prob = 1 - drop_prob
    # shape可以适应多尺度的输入  4dim 5dim
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1) # (b, 1, 1, 1) or (b, 1, 1)
    
    # torch.rand()  => 0~1随机分布的数据
    # “+”  boardcast机制
    random_tensor = keep_prob + torch.rand(shape, dtype=x.type, device=x.device)
     
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return 

import torch.nn as nn
import torch.nn.functional as F

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_lagyer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_lagyer(normalized_shape=4*dim)
        self.reduction = nn.Linear(4*dim, 2*dim , bias=False)
        
    def forward(self, x, H, W):
        
        B, L, C = x.shape
        
        assert L == H * W, 'impout fearture has wrong size'
        
        x = x.view(B, H, W, C)
        
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

            
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4*C)
        
        x = self.norm(x)       # [B, H/2 * W/2, 4*C]
        x = self.reduction(x)  # [B, H/2 * W/2, 2*C]
        
        return x 
    

x = torch.randn((10, 48*48, 3))
lay = PatchMerging(dim = 3)
y = lay(x, H=48, W=48)
print(y.shape)  # [10, 48/2*48/2, 3*2]

