import torch
import torch.nn as nn
import math
# nn.Sequential 自带forward属性

# nn.ModuleList 任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面
# 方法和 Python 自带的 list 一样，无非是 extend，append 等操作,同时 module 的 parameters 也会自动添加到整个网络中。
'''
class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(2)])
    def forward(self, x):
        for m in self.linears:
            x = m(x)
        return x
'''
net1 = nn.Sequential(
    nn.Conv2d(3,6,kernel_size=3,stride=1,padding=1,bias=False),
    nn.BatchNorm2d(num_features=3)
)    

net2_list = [nn.Conv2d(2,4,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(num_features=2)]

net2 = nn.Sequential(*net2_list)   # 直接的list网路层必须要加*

net3 = nn.Sequential(net1, net2)

'''
场景一，有的时候网络中有很多相似或者重复的层，我们一般会考虑用 for 循环来创建它们，而不是一行一行地写，比如：
layers = [nn.Linear(10, 10) for i in range(5)]
场景二，当我们需要之前层的信息的时候，比如 ResNets 中的 shortcut 结构，或者是像 FCN 中用到的 skip architecture 之类的
当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList 比较方便，
'''
print(net1)
'''
    Sequential(
    (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
'''
print(net2)
'''
    Sequential(
    (0): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
'''
ans = 0
for m in net1:
    ans = ans + 1
    print(m)
print(ans)  # 2
'''
    Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
'''
for m in net3: # 可迭代对象，多个sequential时每次都输出一个sequential。 单个sequential时每次都输出一个层
    print(m)
    '''
        Sequential(
            (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )      
        Sequential(
            (0): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
    '''
    if isinstance(m, nn.Conv2d):  # 判断一个对象是否是一个已知的类型
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


net4 = nn.ModuleList([nn.Conv2d(5,10,kernel_size=5,padding=2),nn.BatchNorm2d(5)])
print(net4)
'''
ModuleList(
  (0): Conv2d(5, 10, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (1): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
'''
x = torch.randn((5,5,10,10))
# net4(x)  =>  报错，Module [ModuleList] is missing the required "forward" function

class net5(nn.Module):
    def __init__(self):
        super(net5, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(2)])

    def forward(self, x):
        x = self.linears[0](x)
        y = self.linears[1](x)
        return y

net = net5()
print(net)
'''
net5(
  (linears): ModuleList(
    (0): Linear(in_features=10, out_features=10, bias=True)
    (1): Linear(in_features=10, out_features=10, bias=True)
  )
)
'''
y = net(x)
print(y.shape) # torch.Size([5, 5, 10, 10])