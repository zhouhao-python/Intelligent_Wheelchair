'''
Author: your name
Date: 2021-01-25 12:30:41
LastEditTime: 2021-01-25 13:05:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \modify_module\make_dataset.py
'''
import os

def generate(path):
    image_lists = os.listdir(path)
    with open(path +'\labels.txt', 'a+') as f:
        for img_name in image_lists:
            if img_name.split('_')[0] == 'F':
                label = 1
            elif img_name.split('_')[0] == 'N':
                label = 0
            context = '{path}/{image_name} {label}\n'.format(path=path, image_name=img_name, label=label)
            # filename = os.path.split(file)[0]
            # filetype = os.path.split(file)[1]
            # print(filename, filetype)
            # if filetype == '.txt':
            #     continue
            # name = '/teeth' + '/' + file + ' ' + str(int(label)) + '\n'
            print(context)
            f.write(context)
    print("finished!")


if __name__ == '__main__':
    # 看近处的物体为0 看远处的物体为1
    # img_path = r'D:\研究生期间项目资料\Gaze-Estimation\code\modify_module\dataset\train'
    img_path = r'D:\研究生期间项目资料\Gaze-Estimation\code\modify_module\dataset\val'
    # img_path = r'D:\研究生期间项目资料\Gaze-Estimation\code\modify_module\dataset\test'
    generate(img_path)
