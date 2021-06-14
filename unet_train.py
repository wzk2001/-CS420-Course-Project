from unet_model import UNet 
from torch import optim
import torch.nn as nn
import torch
from unet_loadset import ISBI_Loader

def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
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
    
if __name__ == "__main__":
    # choice the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    data_path = "./aug_dataset/train"
    train_net(net, device, data_path)