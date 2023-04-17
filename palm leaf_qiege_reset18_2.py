# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:57:37 2023

@author: admin
"""

# -*- coding: utf-8 -*-
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):  # 因为漏了这行代码，花了一个多小时解决问题
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

def fetch_dataloaders(data_dir, ratio, batchsize):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)
    character = [[] for i in range(len(dataset.classes))]
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)

    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):
        num_sample_train = int(len(data) * ratio[0])
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val

        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
        for x in data[num_val_index:]:
            test_inputs.append(str(x))
            test_labels.append(i)
    train_transformer_ImageNet = transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        # transforms.Resize(80),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomResizedCrop((80, 80), scale=(0.1, 1), ratio=(0.5, 2)),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64～1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        # transforms.RandomResizedCrop(80, scale=(0.64, 1.0),
        #                                                ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # 标准化图像的每个通道
        transforms.Normalize([0.6941, 0.5584, 0.4225], [0.0749, 0.06599, 0.0570])
        # transforms.Normalize(getStat(train_data))
        ])
    val_transformer_ImageNet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.6941, 0.5584, 0.4225], [0.0749, 0.06599, 0.0570])
        ])

    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, train_transformer_ImageNet), batch_size=batchsize, drop_last=False, shuffle=True)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, val_transformer_ImageNet), batch_size=batchsize, drop_last=False, shuffle=True)
    test_dataloader = DataLoader(MyDataset(test_inputs, test_labels, val_transformer_ImageNet), batch_size=batchsize, shuffle=False)

    loader = {}
    loader['train'] = train_dataloader
    loader['val'] = val_dataloader
    loader['test'] = test_dataloader

    return loader



def get_net():
    num_classes = 2
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss()
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.5f}, '
                f'train acc {metric[1] / metric[2]:.5f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.5f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
#计算数据集的均值标准差，计算完记得传入normalize    
# getStat(ImageFolder(root=r'F:\palmleaf_train\qige_80_color', transform=transforms.ToTensor()))
# ([0.6940837, 0.5584401, 0.42252463], [0.074862584, 0.065944165, 0.05701207])
if __name__ == '__main__':
    data_dir = r'F:\palmleaf_train\qige_80_color'
    """ 每一类图片有3100张，其中780张用于训练，260张用于测试，260张用于测试"""
    loader = fetch_dataloaders(data_dir, [0.8, 0.2, 0], batchsize=256)
    # for x, y in loader['train']:
    #     x
    #     y
    devices, num_epochs, lr, wd = d2l.try_all_gpus(), 60, 0.0001, 0.0005
    lr_period, lr_decay, net = 4, 0.9, get_net()
    train(net, loader['train'], loader['val'], num_epochs, lr, wd, devices, lr_period,
          lr_decay)
