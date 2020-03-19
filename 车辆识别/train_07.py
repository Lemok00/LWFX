import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from mytime import time_change
import time

from torchvision.models import vgg19
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

if __name__ == '__main__':
    TRAIN_DATA_PATH = 'dataset/newtrain/'
    TRAIN_LABEL_PATH = 'dataset/train.csv'
    TRAIN_DATA_RATIO = 0.95
    LEARNING_RATE = 1e-5
    EPOCHS = 40
    PRINT_EPOCH = 1
    VALID_EPOCH = 10
    SAVE_EPOCH = 10
    DOWNLR_EPOCH = 20
    BATCH_SIZE = 16

    # set transformer
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((320, 480)), transforms.RandomCrop((300, 440)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
                                    transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
                                    transforms.ToTensor(), normalize])

    full_dataset = ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
    # split train dataset
    train_size = int(TRAIN_DATA_RATIO * len(full_dataset))
    vaild_size = len(full_dataset) - train_size
    train_dataset, vaild_dataset = random_split(full_dataset, [train_size, vaild_size])

    # set dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=vaild_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # init model
    model = nn.Sequential(vgg19(pretrained=True),nn.Linear(1000, 3), nn.Softmax())
    model.train()
    model.cuda()

    # init optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.CrossEntropyLoss().cuda()

    train_loss_avg = 0.0
    time_start = time.time()
    # train epoch
    for epoch_idx in range(1, EPOCHS + 1):
        # batch-loop
        for batch_idx, data in enumerate(train_loader, start=1):
            # read data
            img, label = data
            img = img.cuda()
            label = label.cuda()

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
            print('[%3d/%3d] loss: %.8f used time: %s rest time: %s'
                  % (epoch_idx, EPOCHS, train_loss_avg / (PRINT_EPOCH * train_size),
                     time_change(time.time() - time_start),
                     time_change((time.time() - time_start) / epoch_idx * (EPOCHS - epoch_idx))))
            sys.stdout.flush()
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
                y = y.numpy()
                y = np.argmax(y, axis=1)
                valid_acc += (label == y).sum()
            print('[VALID] train_acc: %.8f valid_acc: %.8f' % (train_acc / train_size, valid_acc / vaild_size))
            sys.stdout.flush()

        # save_model
        if epoch_idx % SAVE_EPOCH == 0:
            torch.save(model.state_dict(), 'model/0314_vgg19_{}.pth'.format(epoch_idx))

        #
        if epoch_idx % DOWNLR_EPOCH == 0:
            LEARNING_RATE = LEARNING_RATE / 10
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch_idx in range(1, 11):
        # batch-loop
        for batch_idx, data in enumerate(train_loader, start=1):
            # read data
            img, label = data
            img = img.cuda()
            label = label.cuda()

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
        if epoch_idx % 1 == 0:
            print('[%3d/%3d] loss: %.8f used time: %s rest time: %s'
                  % (epoch_idx, EPOCHS, train_loss_avg / (PRINT_EPOCH * train_size),
                     time_change(time.time() - time_start),
                     time_change((time.time() - time_start) / epoch_idx * (EPOCHS - epoch_idx))))
            sys.stdout.flush()
            train_loss_avg = 0.0

        # test model
        if epoch_idx % 1 == 0:
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
                y = y.numpy()
                y = np.argmax(y, axis=1)
                valid_acc += (label == y).sum()
            print('[VALID] train_acc: %.8f valid_acc: %.8f' % (train_acc / train_size, valid_acc / vaild_size))
            sys.stdout.flush()

        # save_model
        if epoch_idx % 1 == 0:
            torch.save(model.state_dict(), 'model/0314_vgg19_b_{}.pth'.format(epoch_idx))
