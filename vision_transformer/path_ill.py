import torch
import torch.nn as nn
from torch.utils import data
from vit_model import VisionTransformer
from torchvision import transforms
import torch.optim as optim


train_data = torch.randn((100, 3, 768, 768))
train_label = torch.rand((100, 2))
for i in range(100):
    if train_label[i][0] >= train_label[i][1]:
        train_label[i][0] = 1
        train_label[i][1] = 0
    else:
        train_label[i][0] = 0
        train_label[i][1] = 1

train_dataset = data.TensorDataset(train_data, train_label)

if __name__ == "__main__":
    
    # 输入数据
    # 定义输入图片预处理的操作
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_loader = data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)
    
    # 224 / 16 = 14
    # 图片大小为768，预训练权重无法加载 768无法被14整除
    # 当然可以设置patch_size = 54 ，但是会丢失部分图像信息
    model = VisionTransformer(img_size=768,
                              patch_size=54,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.train()
    num_epoch = 1

    for epoch in range(num_epoch):
        run_loss = 0.0
        for i, data in enumerate(train_loader):
            input, label = data
            input, label = input.to(device), label.to(device)

            optimizer.zero_grad()

            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()

            if i % 20 == 0:
                print('[{%d}, {%5d}] loss: {%.3f}'.format(epoch+1, i+1, run_loss/20))
                run_loss = 0.0
    
    print('train ok!')
