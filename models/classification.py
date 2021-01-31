import cv2
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Classification(torch.nn.Module):

    def __init__(self):
        super(Classification, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, padding=0)
        # lstm


        # self.conv2 = nn.Conv2d(20, 50, kernel_size=3)
        self.fc1 = nn.Linear(47520, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))   # [N, 20, 24, 99]  N*C*H*W
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # print("Debug", x.shape)  # [N, 20, 24, 99]  N*C*H*W
        x = x.view(-1, 47520)  # 全连接之前需要改变下维度， 变成(N,C*H*W) ,由于batch不是每个循环都一样多，所以使用x.view
        x = F.relu(self.fc1(x))
        x = self.fc2(x)    # [1, 2]
        # print("Debug", x.shape) [47520]
        return x


class CnnLstm(torch.nn.Module):

    def __init__(self):
        super(CnnLstm, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, padding=0)
        # 将变量展开成一维，或者用一个fc层
        self.fc1 = nn.Linear(47520, 512)
        # lstm
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)

        # self.fc2 = nn.Linear(256, 2)  #[N,hidden,classes]
        # self.fc2 = nn.Linear(2560, 2)  #[N,hidden,classes]
        self.fc2 = nn.Linear(2560, 1)  #[N,hidden,classes]

    def forward(self, x_frame_concat):
        cnn_embed_seq = []
        print('debug input size :', x_frame_concat.size())  #[2, 10, 3, 50, 200] 当label.txt中只有一行的时候输出为 [10, 3, 50, 200]，后续就不太对
        assert x_frame_concat.size(1) == 10, 'frame is not expect'
        for seq in range(x_frame_concat.size(1)):
            print(f'seq:{seq}')
            print(x_frame_concat[:,seq, :, :, :],x_frame_concat.size())
            # x = F.relu(F.max_pool2d(self.conv1(x[seq, :, :, :]), (2, 2)))  # [N, 20, 24, 99]  N*C*H*W
            x = F.relu(F.max_pool2d(self.conv1(x_frame_concat[:,seq, :, :, :]), (2, 2)))  # [N, 20, 24, 99]  N*C*H*W
            print(f'conv1.shape: {x.shape}')
            x = x.view(x.size(0), -1)   # [N, 20*24*99= 47520]
            print(f'x.view.shape: {x.shape}')
            x = F.relu(self.fc1(x))     # [N, 512]
            print(f'fc1(x).shape: {x.shape}')
            cnn_embed_seq.append(x)
        print(f'cnn_embed_seq.shape: {len(cnn_embed_seq)}')
        # cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print()
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).permute(1, 0, 2)  # [N ,time_seq, 512]  [2,512,10]
        # cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)  # [N ,time_seq, 512]
        print(f'cnn_embed_seq_T.shape: {cnn_embed_seq.size()}')
        x, (h_n,c_n) = self.lstm(cnn_embed_seq)
        print(f'lstm.shape: {len(x)}')
        print(f'lstm.shape: {x.size()}')
        # x = F.relu(self.fc2(x))

        # x = self.fc2(x.reshape([2, 2560]))
        x = F.sigmoid(self.fc2(x.reshape([-1, 2560])))
        print(f'fc2(x).shape: {x.size()}')
        return x




