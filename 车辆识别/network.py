import torch
import torch.nn as nn

class CarDetectNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #(3,256,256) -> (32,128,128)
        self.conv_1=nn.Sequential(
            nn.Conv2d(3,32,4,2,1),
            nn.ReLU()
        )
        # (32,128,128) -> (64,64,64)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU()
        )
        # (64,64,64) -> (32,32,32)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU()
        )
        # (32,32,32) -> (16,16,16)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 16, 4, 2, 1),
            nn.ReLU()
        )
        # 16*16*16 -> 3
        self.sigm=nn.Sequential(
            nn.Linear(16*16*16,3),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv1=self.conv_1(x)
        conv2=self.conv_2(conv1)
        conv3=self.conv_3(conv2)
        conv4=self.conv_4(conv3)
        y=self.sigm(conv4.view(-1,16*16*16))
        return y