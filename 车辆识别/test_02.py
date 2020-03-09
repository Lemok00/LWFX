import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from torchvision.models import resnet50
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

if __name__ == '__main__':
    TEST_DATA_PATH = 'dataset/test/'
    TEST_MODEL_PATH = 'model/0309_resnet_150.pth'
    RESULT_PATH = 'result/result_0309_resnet.csv'

    # set transformer
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((300, 440)),
                                    transforms.ToTensor(), normalize])

    test_dataset = ImageFolder(root=TEST_DATA_PATH, transform=transform)
    test_size = len(test_dataset)

    # set dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # init model
    # model = CarDetectNetwork()
    model = nn.Sequential(resnet50(pretrained=True), nn.Linear(1000, 3), nn.Softmax())
    model.load_state_dict(torch.load(TEST_MODEL_PATH))
    model.eval()
    model.cuda()

    result = np.zeros(len(test_dataset))
    # batch-loop
    for batch_idx, data in enumerate(test_loader, start=1):
        # read data
        img, _ = data
        img = img.cuda()

        # model forward
        y = model(img).cpu().data
        y = y.numpy()
        y = np.argmax(y, axis=1)
        result[32 * (batch_idx - 1):32 * (batch_idx - 1) + len(y)] = y

    result = result.astype(np.int)
    result_pd = pd.DataFrame(result, index=[i for i in range(0, test_size)])
    result_pd.to_csv(RESULT_PATH, header=None)
