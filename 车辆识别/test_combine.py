import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from torchvision.models import densenet121,resnet50
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

if __name__ == '__main__':
    TEST_DATA_PATH = 'dataset/test/'
    RESULT_PATH = 'result/result_0309_combine03.csv'
    BATCH_SIZE = 16

    # set transformer
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((300, 440)),
                                    transforms.ToTensor(), normalize])

    test_dataset = ImageFolder(root=TEST_DATA_PATH, transform=transform)
    test_size = len(test_dataset)

    # set dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    result = np.zeros(len(test_dataset))
    temp_result=np.zeros(shape=(len(test_dataset),2,3))

    # init model
    # model = CarDetectNetwork()
    model = nn.Sequential(densenet121(pretrained=True),
                          #nn.Linear(1000, 200), nn.ReLU(),
                          nn.Linear(1000, 3), nn.Softmax())
    model.load_state_dict(torch.load('model/0309_densenet_final02.pth'))
    model.eval()
    model.cuda()
    # batch-loop
    for batch_idx, data in enumerate(test_loader, start=1):
        # read data
        img, _ = data
        img = img.cuda()

        # model forward
        y = model(img).cpu().data.numpy()
        temp_result[BATCH_SIZE*(batch_idx-1):BATCH_SIZE*(batch_idx-1)+len(img),0,:]=y
    torch.cuda.empty_cache()

    # init model
    # model = CarDetectNetwork()
    model = nn.Sequential(resnet50(pretrained=True),
                          #nn.Linear(1000, 200), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(1000, 3), nn.Softmax())
    model.load_state_dict(torch.load('model/0309_resnet_final01.pth'))
    model.eval()
    model.cuda()
    # batch-loop
    for batch_idx, data in enumerate(test_loader, start=1):
        # read data
        img, _ = data
        img = img.cuda()

        # model forward
        y = model(img).cpu().data.numpy()
        temp_result[BATCH_SIZE*(batch_idx-1):BATCH_SIZE*(batch_idx-1)+len(img),1,:]=y
    torch.cuda.empty_cache()

    temp_result=temp_result.mean(axis=1)
    result=np.argmax(temp_result,axis=1)
    result_pd = pd.DataFrame(result, index=[i for i in range(0, test_size)])
    result_pd.to_csv(RESULT_PATH, header=None)
