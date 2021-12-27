import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
import matplotlib.pyplot as plt
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import datetime
import torchvision.transforms as transforms
import os

from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

total_loss = []
test_acc = []
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn = nn.Sequential(
            # 单通道输入，6卷积核，卷积核大小为5*5
            nn.Conv2d(1,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第二个卷积层
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 全连接层

        self.fc = nn.Sequential(
            # 特征图大小为120
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            # F6，总共输出84个特征
            nn.Linear(120, 84),
            nn.ReLU(),
            # 转化到分类标签
            nn.Linear(84,10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

def net_train(net, train_data_load, optimizer, epoch, log_interval, loss_func):
    net.train()
    begin = datetime.datetime.now()

    total = len(train_data_load.dataset)
    train_loss = 0 
    correct = 0

    for i, data in enumerate(tqdm(train_data_load)):
        img, label = data
        img, label = img.cuda(), label.cuda()

        optimizer.zero_grad()
        outputs = net(img)
        loss = loss_func(outputs, label)
        loss.backward()
        optimizer.step()
        
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == label).sum()

        if (i+1) % log_interval == 0:
            loss_mean = train_loss / (i+1)
            train_total =  (i + 1) * len(label)
            acc = 100. * correct / train_total
            progress = 100. * train_total / total
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.6f}'.format(
                epoch, train_total, total, progress, loss_mean, acc))
    total_loss.append(train_loss/99)
    end = datetime.datetime.now()
    print('one epoch spend: ', end - begin)

best_acc = 0
def net_test(net, test_data_load, epoch):
    net.eval()
    correct = 0

    for i, data in enumerate(test_data_load):
        img, label = data
        img, label = img.cuda(), label.cuda()
        outputs = net(img)
        _, pre = torch.max(outputs.data, 1)
        correct += (pre == label).sum()

    acc = correct.item() * 100. / len(test_data_load.dataset)
    test_acc.append(acc)
    print('Epoch:{}, ACC:{}\n'.format(epoch,acc))

    global best_acc
    if acc > best_acc:
        best_acc = acc

parser = argparse.ArgumentParser(description='Pytorch CNN training')
parser.add_argument('--epochs',default=100,type=int,help='number of data loading workers')
parser.add_argument('--batch_size',default=512,type=int,help='total batch size of all GPUs on the current node')

def main():
    args = parser.parse_args()

    # load mnist dataset
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=4)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, 
                                            shuffle=False, num_workers=4)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print('Training set size:', len(trainset))
    print('Test set size:',len(testset))
    net = LeNet().cuda()
    # 如果不训练，直接加载保存的网络参数进行测试集验证

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    start_time = datetime.datetime.now()
    test_epochs = []
    for epoch in range(1, args.epochs+1):
        net_train(net, trainloader, optimizer, epoch, log_interval=10, loss_func=nn.CrossEntropyLoss())
        test_epochs.append(epoch)
        # 每个epoch结束后用测试集检查识别准确度
        net_test(net, testloader, epoch)

    end_time = datetime.datetime.now()

    global best_acc
    torch.save(net.state_dict(),'LeNet_MODEL.pth')
    print('MNIST pytorch LeNet Train: EPOCH:{}, BATCH_SZ:{}, LR:{}, ACC:{}'.format(args.epochs, args.batch_size, 0.01, best_acc))
    print('train spend time: ', end_time - start_time)
    # 结果图
    plt.figure(1)
    plt.plot(test_epochs,total_loss,label='train_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('mnist_total_loss.png')

    plt.figure(2)
    plt.plot(test_epochs,test_acc,label='test_acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('mnist_test_acc.png')




if __name__ == '__main__':
    main()