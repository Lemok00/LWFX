import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from imutils import paths
import pandas as pd

class TrainDataFolder(Dataset):
    def __init__(self,folder_path,label_path):
        self.files=sorted(list(paths.list_images(folder_path)))
        self.labels=pd.read_csv(label_path)

    def __getitem__(self, index):
        path=self.files[index%len(self.files)]
        img=np.array(Image.open(path).resize((300,300)))
        #(256,256,3) -> (3,256,256)
        img=np.transpose(img,(2,0,1))
        img=torch.from_numpy(img).float()/255.0

        #get_label
        label=np.zeros(shape=(3))
        img_num=int(path.split("/")[-1].split('.')[0])
        label_num=self.labels['label'][img_num]
        label[label_num]=1
        label=torch.from_numpy(label).float()

        return img,label

    def get_random(self):
        i=np.random.randint(0,len(self.files))
        return self[i]

    def __len__(self):
        return len(self.files)