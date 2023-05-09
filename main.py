import os
from Final_dataset import ToTensor, myDataset
from test import train, test
from Network import Final_Net
import numpy as np
import torch
from torch.utils.data import  DataLoader

if __name__ == 'main':
    datas = {0: 'SH_partA_Density_map', 1: 'SH_partB_Density_map'}
    class_number = {0:22, 1:7}

    for set_idx in range(len(datas)):
        data_dir = datas[set_idx]
        root_dir = os.path.join(r'Test_Data', data_dir)
        class_n = class_number[set_idx]
        label_indice = np.arrange(0.5, class_n + 0.25, 0.5)
        add = np.array([1e-6, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
        label_indice = np.concatenate((add, label_indice))
        label_indice = torch.Tensor(label_indice)
        class_num = len(label_indice) + 1

        img_dir = os.path.join(root_dir, 'test', 'images')
        tar_dir = os.path.join(root_dir, 'test', 'gtdens')
        rgb_dir = os.path.join(root_dir, 'rgbstate.mat')
        # trainset = myDataset(img_dir, tar_dir, rgb_dir, transform=ToTensor(), if_test=False)
        # trainloader = DataLoader(testset, batch_size=1, shuffle=False)
        testset = myDataset(img_dir, tar_dir, rgb_dir, transform=ToTensor(), if_test=True)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)

        net = Final_Net(class_num, label_indice)

        # net = train(net, trainlader)
        mae, mse = test(net, testloader)
        res = '%-10s\t %8.3f\t %8.3f\t ' % ('test', mae, mse) + '\n'
        print(res)