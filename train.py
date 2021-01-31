import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import DataLoader
from option import BasciOption
from models import create_dataloader, Classification

def train(opt):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader = create_dataloader(opt)
    net = Classification()  # 定义训练的网络模型
    net.to(device)
    net.train()
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）

    for epoch in range(opt.num_epochs):  # 一个epoch即对整个训练集进行一次训练
        running_loss = 0.0
        correct = 0
        total = 0
        time_start = time.perf_counter()

        for step, data in enumerate(train_dataloader,
                                    start=0):  # 遍历训练集，step从0开始计算
            inputs, labels = data # 获取训练集的图像和标签
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 清除历史梯度

            # forward + backward + optimize
            # outputs = net(inputs.permute(0,1,3,2))  # 正向传播
            outputs = net(inputs)  # 正向传播
            print('outputs.shape', outputs.shape, labels.shape)
            loss = loss_function(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数
            predict_y = torch.max(outputs, dim=1)[1]
            total += labels.size(0)
            correct += (predict_y == labels).sum().item()
            running_loss += loss.item()
            # print statistics

            # print('train_dataloader length: ', len(train_dataloader))
        acc = correct / total
        print('Train on epoch {}: loss:{}, acc:{}%'.format(epoch + 1, running_loss / total, 100 * correct / total))
        # 保存训练得到的参数
        if opt.model == 'basic':
            save_weight_name = os.path.join(opt.save_path,
                                            'Basic_Epoch_{0}_Accuracy_{1:.2f}.pth'.format(
                                                epoch + 1,
                                                acc))
        elif opt.model == 'plus':
            save_weight_name = os.path.join(opt.save_path,
                                            'Plus_Epoch_{0}_Accuracy_{1:.2f}.pth'.format(
                                                epoch + 1,
                                                acc))
        torch.save(net.state_dict(), save_weight_name)
    print('Finished Training')




if __name__ == '__main__':
    is_train = True
    opt = BasciOption()
    args = opt.initialize(is_train)
    opt.print_args(args)
    train(args)



