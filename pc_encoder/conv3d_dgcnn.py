# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx= knn(x, k=k) 
        else:
            idx= knn(x[:, 6:], k=k)

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


def get_seq_graph_feature(x, k=6, idx=None, dim9=False):
    batch_size = x.size(0)
    frames = x.size(1)
    channel = x.size(2)
    num_points = x.size(3)
    y = torch.zeros(batch_size,frames,channel*2,num_points,k).to(x.device)
    for i in range(batch_size): 
        y[i] = get_graph_feature(x[i], k, idx, dim9)
    y = y.permute(0,2,1,3,4)
    return y.to(x.device)

class conv3d_dgcnn_encoder(nn.Module):
    def __init__(self, args):
        super(conv3d_dgcnn_encoder, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn5 = nn.BatchNorm2d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv3d(3*2, 64, kernel_size=(args.depth,1 +2,1 +2),stride=(args.depth,1 +1,1), bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv3d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv2d(128, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):

        x = get_seq_graph_feature(x, k=self.k)     
        x = self.conv1(x)                       
        x1 = x.max(dim=-1, keepdim=False)[0]   

        x1p = x1.permute(0,2,1,3).to(x.device)   
        x = get_seq_graph_feature(x1p, k=self.k)     
        x = self.conv2(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]    
  

        x = torch.cat((x1, x2), dim=1)  
        x = self.conv5(x)                      
        x = x.max(dim=-1, keepdim=False)[0]   

        x = x.permute(0,2,1)   
        emb = x               
        return emb