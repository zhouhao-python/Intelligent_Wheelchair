'''
Author: your name
Date: 2021-01-25 13:17:53
LastEditTime: 2021-01-25 15:18:16
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \modify_module\option\basic_option.py
'''
import argparse
import os
import icecream as ic

class BasciOption():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self,is_train):
        # self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--batch_size', type=int, default=2)
        self.parser.add_argument('--lr', type=int, default=0.01)
        # self.parser.add_argument('--model', type=str, default='basic')
        self.parser.add_argument('--model', type=str, default='plus')
        # self.parser.add_argument('--device', type=str, default='cuda')
        self.parser.add_argument('--checkpoint_folder', type=str, default='./output')
        self.parser.add_argument('--num_epochs', type=int, default=20)
        self.parser.add_argument('--save_freq', type=int, default=2)

        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--transform', type=bool, default=True)

        # self.parser.add_argument('--train_path', type=str, default=r'D:\研究生期间项目资料\Gaze-Estimation\code\modify_module\dataset\train\labels.txt')
        self.parser.add_argument('--train_path', type=str, default=r'D:\研究生期间项目资料\Smart-wheelchair\code\cnn-lstm\labels.txt')
        self.parser.add_argument('--val_path', type=str, default=r'D:\研究生期间项目资料\Gaze-Estimation\code\modify_module\dataset\val\labels.txt')

        if is_train == True:
            file_name = os.listdir('./')
            # ic(file_name)
            # print(file_name)
            if 'output' not in file_name:
                os.makedirs('./output/train/weights')
            else:
                if 'weights' not in os.listdir('./output/train'):
                    os.mkdir('./output/train/weights')
            exp_times = os.listdir('./output/train/weights')
            os.makedirs(f'./output/train/weights/exp_{len(exp_times)+1}')
            self.parser.add_argument('--is_train', type=bool, default=True)
            self.parser.add_argument('--save_path', type=str,
                                 default='./output/train/weights/exp_{}'.format(len(exp_times)+1))
        else:

            # test option

            self.parser.add_argument('--test_path', type=str,
                                     default=r'D:\研究生期间项目资料\Gaze-Estimation\code\modify_module\dataset\test\labels.txt')
            self.parser.add_argument('--test_batch_size', type=int, default=1)
            file_name = os.listdir('./')
            # ic(file_name)
            # print(file_name)
            if 'output' not in file_name:
                os.makedirs('./output/predict')
            else:
                if 'predict' not in os.listdir('./output'):
                    os.mkdir('./output/predict')
            exp_times = os.listdir('./output/predict')
            os.makedirs(f'./output/predict/exp_{len(exp_times) + 1}')
            self.parser.add_argument('--is_train', type=bool, default=False)

        args = self.parser.parse_args()
        return args

    def print_args(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            # default = self.parser.get_default(k)
            # if v != default:
            #     comment = '\t[default: %s]' % str(default)
            # message += '{:>20}: {:<30}{}\n'.format(str(k), str(v), comment)
            message += f'{str(k):>20}: {str(v):<30}{comment}\n'
        message += '----------------- End -------------------'
        print(message)

if __name__ == '__main__':
    opt = BasciOption()
    args = opt.initialize()
    opt.print_args(args)
