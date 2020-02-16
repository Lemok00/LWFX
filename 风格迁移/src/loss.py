import torch
import torch.nn as nn


# 计算内容损失
class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        # Loss = MSE(input * weight, target * weight)
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


# Gram矩阵
class GramMatrix(nn.Module):
    def forward(self, input):
        # a=1 b=特征图数
        # c=w d=h
        a, b, c, d = input.size()
        # reshape
        features = input.view(a * b, c * d)
        # 矩阵相乘
        # 计算Gram矩阵
        Gram = torch.mm(features, features.t())
        # 返回归一化的结果
        return Gram.div(a * b * c * d)


# 计算风格损失
class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
