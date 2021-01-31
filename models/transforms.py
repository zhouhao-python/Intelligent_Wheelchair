'''
Author: your name
Date: 2021-01-25 14:27:44
LastEditTime: 2021-01-25 15:10:26
LastEditors: your name
Description: In User Settings Edit
FilePath: \modify_module\models\transforms.py
'''
'''
Author: your name
Date: 2021-01-25 14:27:44
LastEditTime: 2021-01-25 14:27:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \modify_module\models\transforms.py
'''
from torchvision import transforms
from PIL import Image

def create_transform(opt):
    if opt.transform == True:
        transform = transforms.Compose([
            transforms.Resize((50, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform
    