import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model  创建对应的model
    # has_logits若在训练时，设置为true，预测时必须也设置为True
    # 决定了其权重文件包含head -> 1/2个Linear层
    model = create_model(num_classes=5, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    
    # torch.load()表示加载已经训练好的模型 
    # 1 - torch.save(model,PATH) 
    # 1 - model_new = torch.load(PATH)
    # 2 - torch.save(model.state_dict(), PATH)
    # 2 - model_new = net()
    # 2 - model_new.load_state_dict(torch.load(PATH), map_location="cuda:0")
    # ||| 首先读取预训练模型参数，然后加载预训练模型的参数
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    # 取消有些参数沿着梯度下降进行更新
    # output/predict/predict_cla => requires_grad=False
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
