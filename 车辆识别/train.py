import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataloader import TrainDataFolder
#from network import CarDetectNetwork
from torchvision.models import Inception3

if __name__ == '__main__':
    TRAIN_DATA_PATH = 'dataset/train/'
    TRAIN_LABEL_PATH = 'dataset/train.csv'
    TRAIN_DATA_RATIO = 0.8
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    PRINT_EPOCH = 1
    VALID_EPOCH = 10

    full_dataset = TrainDataFolder(TRAIN_DATA_PATH,TRAIN_LABEL_PATH)
    # split train dataset
    train_size = int(TRAIN_DATA_RATIO * len(full_dataset))
    vaild_size = len(full_dataset) - train_size
    train_dataset, vaild_dataset = random_split(full_dataset, [train_size, vaild_size])

    # set dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=vaild_dataset, batch_size=32, shuffle=False)

    # init model
    #model = CarDetectNetwork()
    model=nn.Sequential(Inception3(num_classes=3,aux_logits=False),nn.Sigmoid())
    model.train()
    model.cuda()

    # init optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.BCEWithLogitsLoss()

    train_loss_avg = 0.0
    # train epoch
    for epoch_idx in range(1, EPOCHS + 1):
        # batch-loop
        for batch_idx, data in enumerate(train_loader, start=1):
            # read data
            img, label = data
            img = img.cuda()
            label=label.cuda()

            # model forward
            y = model(img)

            # compute loss
            loss = loss_criterion(y, label)
            train_loss_avg += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print infomation
        if epoch_idx % PRINT_EPOCH == 0:
            print('[%3d/%3d] loss: %.8f' % (epoch_idx, EPOCHS, train_loss_avg / (PRINT_EPOCH * train_size)))
            train_loss_avg = 0.0

        # test model
        if epoch_idx % VALID_EPOCH == 0:
            # test train data
            train_acc, valid_acc = 0.0, 0.0
            for batch_idx, data in enumerate(train_loader, start=1):
                # read data
                img, label = data
                img = img.cuda()
                # model forward
                y = model(img).cpu().data
                # compute accuracy
                label = label.numpy()
                label = np.argmax(label, axis=1)
                y = y.numpy()
                y = np.argmax(y, axis=1)
                train_acc += (label == y).sum()
            # test valid data
            for batch_idx, data in enumerate(valid_loader, start=1):
                # read data
                img, label = data
                img = img.cuda()
                # model forward
                y = model(img).cpu().data
                # compute accuracy
                label = label.numpy()
                label = np.argmax(label, axis=1)
                y = y.numpy()
                y = np.argmax(y, axis=1)
                valid_acc += (label == y).sum()
            print('[VALID] train_acc: %.8f valid_acc: %.8f' % (train_acc / train_size, valid_acc / vaild_size))
