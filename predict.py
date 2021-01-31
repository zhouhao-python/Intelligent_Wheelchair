'''
Author: your name
Date: 2021-01-25 12:30:27
LastEditTime: 2021-01-25 15:14:26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \modify_module\predict.py
'''
import torch
import os
from PIL import Image
# from torchvision import transforms
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from models import create_dataloader,Classification
from option import BasciOption
classes = ('0', '1')

device = torch.device('cuda')
# device = torch.device('cpu')






'''
def predict(img_path):
    # net=torch.load('Lenet.pth') # pth格式 只保留参数
    # print('net', net)

    net = models.Classification()
    net.load_state_dict(torch.load('Lenet.pth'))
    net = net.to(device)
    torch.no_grad()
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('this picture maybe :', classes[predicted[0]])
    return classes[predicted[0]]
'''


def predict(opt):
    # net=torch.load('Lenet.pth') # pth格式 只保留参数
    # print('net', net)
    '''
    需要将basic_option中的is_train 修改为false
    '''
    # opt.is_train = False
    acc = 0
    total = 0
    test_dataloader = create_dataloader(opt)
    net = Classification()
    net.load_state_dict(torch.load(f"./output/train/weights/exp_1/Basic_Epoch_20_Accuracy_0.99.pth"))
    net = net.to(device)
    with torch.no_grad():
        for index, data in enumerate(test_dataloader,start=1):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            print(f'number {index} picture maybe : {classes[predicted[0]]}')
            total += labels.size(0)
            acc += (predicted == labels).sum().item()
    print('Accuracy on test set : {}%'.format(100 * acc / total))




if __name__ == '__main__':
    # predict(r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\dataset\test\N_both_eyes_area_frame_48.jpg')
    # predict(r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\result\images\test\video2_far\concat\F_both_eyes_area_frame_300.jpg')
    # predict(r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\result\images\test\video2_near\concat\N_both_eyes_area_frame_30.jpg')
    '''
    result = []
    c = 0
    for i in range(300,501):
        if os.path.exists(r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\result\images\test\video2_far\concat\F_both_eyes_area_frame_{}.jpg'.format(i)):
            c += 1
            result.append(predict(
                r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\result\images\test\video2_far\concat\F_both_eyes_area_frame_{}.jpg'.format(i)))
        else:
            print('pic_{} is not exit'.format(i))
    print('acc = ', result.count('1')/c)  # 测试准确率为0.7846153846153846
    '''

    is_train = False
    opt = BasciOption()
    args = opt.initialize(is_train)
    opt.print_args(args)
    # print(opt)
    predict(args)
