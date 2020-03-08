import os

for i in range(0, 3450):
    src_path = 'dataset/test/%d.jpg' % i
    tar_path = 'dataset/test/%04d.jpg' % i
    os.rename(src_path, tar_path)

import os
import pandas as pd
from imutils import paths
from PIL import Image

for i in range(3):
    os.makedirs('dataset/newtrain/%d' % i)
labels = pd.read_csv('dataset/train.csv')
imgs = paths.list_images('dataset/train/')
for img in imgs:
    img_num = int(img.split("/")[-1].split('.')[0])
    label_num = labels['label'][img_num]
    img = Image.open(img).resize((300, 300))
    img.save('dataset/newtrain/%d/%04d.png' % (label_num, img_num))
