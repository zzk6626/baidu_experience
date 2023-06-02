import torch
import torch.nn as nn

class net(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.block1 = nn.Sequential(nn.Conv2d(3,3,kernel_size=5),
                                   nn.BatchNorm2d(num_features=3))
        
        self.block2 = nn.Sequential(nn.Conv2d(3,3,kernel_size=5),
                                   nn.BatchNorm2d(num_features=3))
        
        # 权重初始化方法一  
        self.apply(self.weight_init)

        # 权重初始化方法二
        # self.apply(weight_init)
    
    # 权重初始化方法一 - 类内定义权重初始化方法
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)  # 截断正态分布
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # kaiming正态分布，一般用来初始化卷积操作
            nn.init.kaiming_normal_(m.weight, mode="fan_out") 
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        
# 权重初始化方法二 - 在类外定义weight_init方法
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)  # 截断正态分布
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # kaiming正态分布，一般用来初始化卷积操作
        nn.init.kaiming_normal_(m.weight, mode="fan_out") 
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# 权重初始化方式三 - 创建对象后初始化权重
net = net()
for m in net.modules():
    # 1-先输出一个完整的net 包含net(block1,block2)
    # 2-再输出单纯的block1  Sequential(Conv2d, BatchNorm2d)
    # 3-输出block1里面的详细层 Conv2d  BatchNorm2d
    # 4-重复步骤(2-3)，输出block2相关信息
    print(m)   
    if isinstance(m, nn.Conv2d):
        print(m.weight.data.shape) # torch.Size([3, 3, 5, 5])
        m.weight.data.normal_(0, 1)
    if isinstance(m, nn.BatchNorm2d):
        print(m.weight.data.shape) # torch.Size([3])
        m.weight.data.fill_(1)
        m.bias.data.fill_(0) # OR zero_()
        
