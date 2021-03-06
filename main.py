from PIL.Image import LINEAR
from unet_model import UNet 
from torch import optim
import torch.nn as nn
import torch
from unet_loadset import ISBI_Loader
import glob
import numpy as np
import torch
import os
import cv2
from unet_accmy import*

# Train

def train_net(net, device, data_path, epochs, batch_size, lr):
    #load train set
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    
    # RMSprop 
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # lOSS
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    #train
    for epoch in range(epochs):
        net.train()
        # 让迭代的后1/4部分学习率降低
        
        if epoch >= 0.6 * epochs:
            lr = 0.5 * lr
        if epoch >=0.2 * epochs:
           lr = 0.5 *lr
        
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            #output
            pred = net(image)
            #caculate the loss
            loss = criterion(pred, label)
            print('Loss/train',loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # update the parameters
            loss.backward()
            optimizer.step()

#设置三个关键参数
epochs = 20
batch_size=2
lr=0.00001   

if __name__ == "__main__":
    # choice the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    data_path = "./aug_dataset/train"
    train_net(net, device, data_path,epochs,batch_size,lr)


#Test

import numpy as np
import torch
import os
import cv2
from unet_model import UNet

#创建一个output文件夹存放测试出的图片
path = os.getcwd()
filepath = path +'/aug_dataset/test'+'\\'+'output'
os.mkdir(filepath)
savepath = path +'/aug_dataset/test/output/'

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('./aug_dataset/test/image/*')
    # 遍历所有图片
    n = 0
    for test_path in tests_path:
        # 保存结果地址
        save_res_path = savepath + str(n)+'.png'
        n = n + 1
        
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)


# compute acc

unet_acc(savepath,'./aug_dataset/test/label/')
print('epochs=',epochs,0.5*epochs,0.25*epochs)
print('batch_size=',batch_size)
print('lr=',lr)

import winsound
winsound.Beep(500,1000)