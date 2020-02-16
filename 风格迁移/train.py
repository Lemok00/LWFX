import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import numpy as np

from torchvision import models
from torchsummary import summary

from src.vgg_model import get_style_model_and_losses
from src.mytime import time_change

import time

'''
读取图片
'''

style_img_path = './dataset/style001.jpg'
content_img_path = './dataset/content001.jpg'

style_img = Image.open(style_img_path).convert('RGB')
content_img = Image.open(content_img_path).convert('RGB')

style_img_W, style_img_H = style_img.size
content_img_W, content_img_H = content_img.size

style_img = np.array(style_img.resize((512, 512))).astype(np.float) / 255.0
content_img = np.array(content_img.resize((512, 512))).astype(np.float) / 255.0

style_img = np.transpose(style_img, (2, 0, 1))
content_img = np.transpose(content_img, (2, 0, 1))

style_x = torch.from_numpy(style_img).type(torch.FloatTensor).unsqueeze(0).cuda()
content_x = torch.from_numpy(content_img).type(torch.FloatTensor).unsqueeze(0).cuda()
input_x = content_x.clone()

'''
设置参数
'''

STYLE_WEIGHT = 1000
CONTENT_WEIGHT = 1
MAX_EPOCH = 300
PRINT_EPOCH = 50

'''
加载网络
'''

model, style_losses, content_losses = get_style_model_and_losses \
    (style_img=style_x, content_img=content_x, style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT)
input_pram = nn.Parameter(input_x.data)
optimizer = optim.LBFGS([input_pram])

# summary(model,input_size=(3,512,512))

'''
训练网络
'''
start = time.time()
for epoch_idx in range(1, MAX_EPOCH + 1):
    def closure():
        input_pram.data.clamp(0, 1)

        optimizer.zero_grad()
        model(input_pram)
        style_score, content_score = 0.0, 0.0

        for sl in style_losses:
            style_score += sl.backward()
        for cl in content_losses:
            content_score += cl.backward()

        if epoch_idx % PRINT_EPOCH == 0:
            now = time.time()
            print(
                '[%03d/%03d] Style Loss: %.4f Content Loss: %.4f' % (
                    epoch_idx, MAX_EPOCH, style_score, content_score))
            print('Used Time: %s Rest Time: %s' % (
                time_change(now - start), time_change((now - start) / epoch_idx * (MAX_EPOCH - epoch_idx))))
        return style_score + content_score


    optimizer.step(closure)
input_pram.data.clamp(0, 1)

output_img = np.array(input_pram.cpu().data)[0] * 255.0
output_img = np.transpose(output_img, (1, 2, 0)).astype(np.uint8)
output_img = Image.fromarray(output_img).resize((content_img_W, content_img_H))
output_img.save('./result/result001.png')
