#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao, Ziyi Wu
@Contact: ta19@mails.tsinghua.edu.cn, dazitu616@gmail.com
@Time: 2022/7/30 7:49 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class FullSelfAttention(nn.Module):
    def __init__(self, dim,k):
        super().__init__()
        self.k = k
        self.dim = dim
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        # q*k^T/sqrt(d_k)
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention / (self.dim ** 0.5)
        attention = F.softmax(attention, dim=-1)
        output = torch.bmm(attention, v)
        idx = attention.topk(k=self.k, dim=-1,largest=True)[1]
        return idx, output.transpose(1, 2)


class conv2d_sa_dgcnn_encoder(nn.Module):
    def __init__(self, args):
        super(conv2d_sa_dgcnn_encoder, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64+64)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=(3,3),stride=(2,1), bias=False), 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64*2, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(128+64, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.sa1 = FullSelfAttention(3,self.k)
        self.sa2 = FullSelfAttention(64,self.k)


    def forward(self, x):
        batch_size = x.size(0)
        x = self.get_graph_feature(x, k=self.k,layer=0)      
        x = self.conv1(x)                       
        x1 = x.max(dim=-1, keepdim=False)[0]    

        x = self.get_graph_feature(x1, k=self.k,layer=1)     
        x = self.conv2(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]    

        x = torch.cat((x1, x2), dim=1)         
        x = self.conv5(x)                      

        x = x.max(dim=-1, keepdim=False)[0]    
        feat = x    # global feature

        return feat

    def knn(self,x, k):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  
        return idx


    def get_graph_feature(self,x, k=6, idx=None, dim9=False,layer=0):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            if layer == 0:
                idx,x_sa = self.sa1(x.transpose(2,1))
            elif layer == 1:
                idx,x_sa = self.sa2(x.transpose(2,1))

        # residual connection
        x = x+x_sa

        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()  
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims) 
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
        return feature    