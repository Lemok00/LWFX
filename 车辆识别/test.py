import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataloader import TestDataFolder
#from network import CarDetectNetwork
from torchvision.models import Inception3

if __name__ == '__main__':
    TEST_DATA_PATH = 'dataset/test/'
    TEST_MODEL_PATH='model/inception_700.pth'
    RESULT_PATH='result/result_0308.csv'

    test_dataset = TestDataFolder(TEST_DATA_PATH)
    test_size=len(test_dataset)

    # set dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # init model
    #model = CarDetectNetwork()
    model=nn.Sequential(Inception3(num_classes=3,aux_logits=False),nn.Sigmoid())
    model.load_state_dict(torch.load(TEST_MODEL_PATH))
    model.eval()
    model.cuda()

    result=np.zeros(len(test_dataset))
    # batch-loop
    for batch_idx, data in enumerate(test_loader, start=1):
        # read data
        img = data
        img = img.cuda()

        # model forward
        y = model(img).cpu().data
        y = y.numpy()
        y = np.argmax(y, axis=1)
        result[32*(batch_idx-1):32*(batch_idx-1)+len(y)]=y

    result=result.astype(np.int)
    result_pd=pd.DataFrame(result,index=[i for i in range(0,test_size)])
    result_pd.to_csv(RESULT_PATH,header=None)