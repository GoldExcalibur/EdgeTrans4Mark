from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import dirname, join, abspath

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import namedtuple

class ResNet18(nn.Module):
    def __init__(self, pretrained, requires_grad=True):
        super().__init__()
        res18 = models.__dict__['resnet18'](pretrained=False)
        if pretrained:
            state_path = ''
            res18.load_state_dict(torch.load(state_path))
        for n in ['conv1', 'bn1', 'relu', 'maxpool', \
            'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            m = getattr(res18, n)
            setattr(self, n, m)
        self.final_inp_channels = 512
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False, \
            names=['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']):
        super(Vgg16, self).__init__()
        self.names = names
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = vgg_pretrained_features[:3] #conv1_2
        self.slice2 = vgg_pretrained_features[3:8] #conv2_2
        self.slice3 = vgg_pretrained_features[8:13] #conv3_2
        self.slice4 = vgg_pretrained_features[13:20] #conv4_2
        self.slice5 = vgg_pretrained_features[20:27] #conv5_2
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        feats = []
        h = self.slice1(X)
        feats.append(h)
        h = self.slice2(h)
        feats.append(h)
        h = self.slice3(h)
        feats.append(h)
        h = self.slice4(h)
        feats.append(h)
        h = self.slice5(h)
        feats.append(h)

        out = dict(zip(self.names, feats))
        return out