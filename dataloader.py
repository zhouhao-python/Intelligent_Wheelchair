'''
Author: your name
Date: 2021-01-25 14:27:07
LastEditTime: 2021-01-25 15:37:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \modify_module\models\dataloader.py
'''
from torch.utils.data import DataLoader
from models.mydataset import MyDataset, CnnLstmDataset
from models.transforms import create_transform
from option import BasciOption

def create_dataloader(opt):
    if opt.model == 'basic':
        if opt.is_train == True:
            train_dataset = MyDataset(txt_path=opt.train_path, transform=create_transform(opt))
            val_dataset = MyDataset(txt_path=opt.val_path,
                                    transform=create_transform(opt))

            train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.num_workers)

            val_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        num_workers=opt.num_workers)
            return train_dataloader, val_dataloader
        else:
            test_dataset = MyDataset(txt_path=opt.test_path, transform=create_transform(opt))

            test_dataloader = DataLoader(dataset=test_dataset,
                                        batch_size=opt.test_batch_size,
                                        shuffle=False,
                                        num_workers=opt.num_workers)
            print('is_train', opt.is_train)
            return test_dataloader
    elif opt.model == 'plus':
        train_dataset = CnnLstmDataset(txt_path=opt.train_path,
                                  transform=create_transform(opt))
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      num_workers=opt.num_workers)
        return train_dataloader

if __name__=="__create_dataloader__":
    opt = BasciOption().initialize()
    create_dataloader(opt)