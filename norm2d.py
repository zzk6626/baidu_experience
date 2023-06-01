import torch
import torch.nn as nn

data = torch.randn((1, 3, 64, 64))

bn_net = nn.BatchNorm2d(num_features=3)
'''
BatchNorm1d  => 所有BN操作后 tensor的shape保持不变
xnew = (1-momentum) * x_cur + momentum * x_batch  => 每个channel均值和方差都会更新
data_new = (data_old - x_mean) / sqrt(var+eps)
print(m.running_mean)   =>  全局均值
print(m.running_var)    =>  全局方差
我们提前设定的num_features时不一样的,当做BN的tensor维度是(N,C,L)时,我们定义的num_features是C
意味着我们会根据每个通道的不同样本的L长度特征进行相加再除以N*L得到均值,因此会得到C个均值。
再具体点的实例就是输入为(5,3,10)的tensor,我们会取[0,0,:],[1,0,:],....[4,0,:]
这些向量的数值(6一共有5*10个数字)加起来除以5*10得到第一个通道的均值,并对以上数字进行正则化。
当做BN的tensor维度是(N,L),我们定义的num_features是L,因此我们会计算出L个均值和方差,可以看成(N,L,1)的形式,每一个特征长度为1,只有L个通道
具体点的实例:输入维度为(4,5)的tensor,会取[0,0],[1,0],[2,0],[3,0]这4个数进行正则化,可以知道我们最终会得到L个均值和方差
'''
ln_net = nn.LayerNorm([3,64,64])   # or nn.LayerNorm([64,64])
'''
normalized_shape:可以设定为:int,列表,或者torch.Size([3, 4])
eps:对输入数据进行归一化时加在分母上,防止除零。
elementwise_affine:是否进行仿射变换,如果是True则此模块具有可学习的仿射参数weight和bias,使得能够使得数据不仅仅是服从N(0,1)正态分布。
如果normalized_shape传入的是列表,比如[3,4],那么需要要求传入的tensor需要最后两个维度需要满足[3, 4],会把最后两个维度以用12个数据
进行求均值和方差并正则化。具体一点的例子,传入的tensor维度为(N,C,3,4)那么会对【0,0,:,:】
这12个数进行正则化,【0,1,:,:】这12个数进行正则化.....因此最后得到会得到N*C个均值和方差。看例子
'''
out1 = bn_net(data)
out2 = ln_net(data)

print(out1.shape)   # ([1, 3, 64, 64])
print(bn_net.running_mean)
print(bn_net.running_var)
print(out2.shape)   # ([1, 3, 64, 64])
print(ln_net.weight)
print(ln_net.bias)
'''
使用特点:
1- BN抹平了不同特征之间的大小关系，而保留了不同样本之间的大小关系。这样，如果具体任务依赖于不同样本之间的关系，BN更有效，尤其是在CV领域，例如不同图片样本进行分类，不同样本之间的大小关系得以保留。
2- LN抹平了不同样本之间的大小关系，而保留了不同特征之间的大小关系。所以，LN更适合NLP领域的任务，其中，一个样本的特征实际上就是不同word embedding，通过LN可以保留特征之间的这种时序关系。

BN使featu1、featu2....服从norm分布 - 抹平了不同特征之间的大小关系
LN使batch1、batch2....服从norm分布 - 抹平了不同样本之间的大小关系

img: https://img-blog.csdnimg.cn/95f3405f30c9421d9a5a708a6418381b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5pWy5Luj56CB55qEcXVhbnQ=,size_10,color_FFFFFF,t_70,g_se,x_16
'''
