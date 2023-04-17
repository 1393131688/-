# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:21:32 2023

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:27:16 2023

@author: admin
"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
batch_size = 256


 
 
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
 
 
if __name__ == '__main__':
    train_dataset = ImageFolder(root=r'./data/food/', transform=None)
    print(getStat(train_dataset))
#图像增广 
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
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                                      [0.2023, 0.1994, 0.2010])
    ])
val_transformer_ImageNet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                                      [0.2023, 0.1994, 0.2010])
    ])
class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]
 
def split_Train_Val_Data(data_dir, ratio):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)     # data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    #print(dataset.class_to_idx)
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)
    #print(dataset.samples)
 
    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):   # data为一类图片
        num_sample_train = int(len(data) * ratio[0])
        #print(num_sample_train)
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val
        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
    #print(len(train_inputs))
    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, train_transformer_ImageNet),
                                  batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, val_transformer_ImageNet),
                                  batch_size=batch_size, shuffle=False)
 
    return train_dataloader, val_dataloader
 
 
def data_loader(dataset_dir, batch_size):
    img_data = ImageFolder(dataset_dir,transform=transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]))
    data_loader = DataLoader(img_data, batch_size=batch_size, shuffle=True)
 
    return data_loader
 


data_dir = r'F:\palmleaf_train\qige_80_color'
train_iter, test_iter = split_Train_Val_Data(data_dir, [1, 0])

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

X = torch.rand(size=(1, 3, 80, 80))
for layer in get_net():
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
   
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 60, 0.0005, 0.001
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, test_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)