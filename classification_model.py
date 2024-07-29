# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pc_encoder.pointnet import pointnet_encoder
from pc_encoder.pn2 import PointNet2Encoder
from pc_encoder.pointmlp import PointMLPEncoder

from pc_encoder.conv2d_dgcnn import DGCNN_encoder as conv2d_dgcnn_encoder
from pc_encoder.conv2d_sa_dgcnn import conv2d_sa_dgcnn_encoder
from pc_encoder.conv3d_dgcnn import conv3d_dgcnn_encoder
from pc_encoder.conv3d_sa_dgcnn import conv3d_sa_dgcnn_encoder

# Transformer
from torch.nn import TransformerEncoder,TransformerEncoderLayer
import math
from timm.models.layers import trunc_normal_
# mort sort
from utils.mort_sort import simplied_morton_sorting,morton_sorting
# rnn sort
from utils.pointcloudRnn import PointCloudSortingRNN


class pcseq_classifier(nn.Module):
    def __init__(self, args):
        super(pcseq_classifier, self).__init__()
        self.args = args
        self.emb_dims = args.emb_dims
        self.hidden_dims = args.hidden_dims
        self.what_encoder = args.encoder
        device = torch.device("cuda" if args.cuda else "cpu")
        if self.what_encoder == 'conv2d_dgcnn':
            self.encoder = conv2d_dgcnn_encoder(args).to(device) 
        elif self.what_encoder == 'conv2d_sa_dgcnn':
            self.encoder = conv2d_sa_dgcnn_encoder(args).to(device)
        elif self.what_encoder == 'conv3d_dgcnn':
            self.encoder = conv3d_dgcnn_encoder(args).to(device)
        elif self.what_encoder == 'conv3d_sa_dgcnn':
            self.encoder = conv3d_sa_dgcnn_encoder(args).to(device)
        elif self.what_encoder == 'pointnet':
            self.encoder = pointnet_encoder(args).to(device)
        elif self.what_encoder == 'pointnet2':
            self.encoder = PointNet2Encoder(args).to(device)
        elif self.what_encoder == 'pointmlp':
            self.encoder = PointMLPEncoder(args,k_neighbors=[6],dim_expansion=[4],pre_blocks=[1],pos_blocks=[1],reducers=[2],normalize="center",res_expansion=0.5).to(device)
        
        # transformer, wait to be optimized
        transformer_encoder_layer = TransformerEncoderLayer(d_model=args.emb_dims, nhead=4, dim_feedforward=256, dropout=args.dropout, activation='relu')
        self.transformer = TransformerEncoder(transformer_encoder_layer, num_layers=2)
        # add cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dims))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.emb_dims))
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        #use transformer
        self.fc1 = nn.Linear(args.hidden_dims*2 *2, args.hidden_dims*2) 
        self.fc1_5 = nn.Linear(args.hidden_dims*2, args.hidden_dims)
        self.fc2 = nn.Linear(args.hidden_dims, int(args.hidden_dims/2))
        self.bn1 = nn.BatchNorm1d(args.hidden_dims*2)
        self.bn1_5 = nn.BatchNorm1d(args.hidden_dims)
        self.bn2 = nn.BatchNorm1d(int(args.hidden_dims/2))
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(int(args.hidden_dims/2), args.num_classes)

        # sort_method
        self.sort = PointCloudSortingRNN(128, 256) if args.sort == 'rnn' else None
        self.batch_sort = self.batch_pcseq_rnn if args.sort == 'rnn' else self.batch_pcseq_mort
    
    def generate_positional_encoding(self, seq_len, d_model):
        PE = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        return PE
        
    def forward(self, x,frame_length):
        x = self.batch_sort(x,frame_length)
        x=x.permute(0,1,3,2)
        batch_sz = x.size(0)

        if '3d' not in self.what_encoder:
            x = x.permute(1,0,2,3)
            index_seq = np.argsort(frame_length.cpu().numpy()).tolist()
            x = x[:,index_seq]
            ascending_frame_length = sorted(frame_length)
            reverse_to_original_index = np.argsort(index_seq)
            y = torch.zeros((max(frame_length), batch_sz, self.emb_dims), device=x.device)
            if self.what_encoder in ['conv2d_sa_dgcnn','conv2d_dgcnn','pointnet2','pointmlp','pointnet']:
                head = 0
                for m in range(max(frame_length)):
                    while(m>=ascending_frame_length[head]):
                        head += 1
                    y[m][head:]= self.encoder(x[m][head:].type(torch.FloatTensor).to(x.device))
                y = y[:,reverse_to_original_index]
            y = y.permute(1,0,2)
        else:
            y = self.encoder(x) #y: (batch_size,frame_num,emb_dims)

        # Transformer part
        seq_len = y.size(1)  # Assuming y is of shape (batch_size, seq_len, d_model)
        cls_tokens = self.cls_token.expand(batch_sz, -1, -1)
        cls_pos = self.cls_pos
        pos_encoding = self.generate_positional_encoding(seq_len, self.emb_dims).unsqueeze(0).to(y.device)
        y = torch.cat((cls_tokens, y), dim=1)
        pos_encoding = torch.cat((cls_pos, pos_encoding), dim=1)
        y = y + pos_encoding
        y = y.permute(1,0,2)
        lengths = frame_length+1
        src_key_padding_mask = (torch.arange(seq_len+1).to(x.device) >= lengths.unsqueeze(1)).to(x.device) #src_key_padding_mask: (batch_size,seq_len')
        y = self.transformer(y,src_key_padding_mask=src_key_padding_mask) 
        y = y.permute(1,0,2) 
        # 取cls_token对应的输出作为整体序列的特征表示的一部分
        z1 = y[:,0,:].to(x.device)
        # 取最后一个有效frame对应的输出作为整体序列的特征表示的一部分/有效frame中最大值
        z2 = torch.zeros(batch_sz,self.emb_dims).to(x.device)
        for i in range(batch_sz):
            z2[i] = y[i,frame_length[i],:]
        z = torch.cat((z1,z2),dim=1)

        z = self.fc1(z) 
        y1 = self.bn1(z)  
        y1 = F.leaky_relu(y1,negative_slope=0.2)
        y1 = self.dropout(y1) 

        # add for use transformer
        y1 = self.fc1_5(y1)
        y1 = self.bn1_5(y1)
        y1 = F.leaky_relu(y1,negative_slope=0.2)
        y1 = self.dropout(y1)

        y1 = self.fc2(y1)  
        y1 = self.bn2(y1)   
        y1 = F.leaky_relu(y1,negative_slope=0.2)
        y1 = self.dropout(y1)

        y1 = self.fc(y1)
        return y1
    

    def batch_pcseq_mort(self,pcseq,frame_length):
        b, f, n, c = pcseq.size()
        for i in range(b):
            #sorted_indices = simplied_morton_sorting(pcseq[i,:frame_length[i],:,:])
            sorted_indices = morton_sorting(pcseq[i,:frame_length[i],:,:])
            pcseq[i,:frame_length[i],:,:] = pcseq[i,:frame_length[i],:,:].view(-1,3)[sorted_indices,:].view(frame_length[i],n,c)
        return pcseq
    
    def batch_pcseq_rnn(self,pcseq,frame_length):
        b, f, n, c = pcseq.size()
        pcseq = pcseq.view(-1,n,c)
        id_base = torch.arange(0, b, device=pcseq.device) * f

        indices = torch.cat([id_base[i] + torch.arange(frame_length[i], device=pcseq.device) for i in range(b)])

        assert indices.size(0) == torch.sum(frame_length).item()

        pcseq2 = torch.index_select(pcseq, 0, indices) # (indices.size(0),n,c)

        sorted_indices = self.sort(pcseq2)  # (indices.size(0),n)

        pcseq2 = pcseq2.gather(1, sorted_indices.unsqueeze(2).expand(-1, -1, pcseq2.size(2)))

        pcseq.index_copy_(0, indices, pcseq2.view(-1, n, c))
        pcseq = pcseq.view(b,f,n,c)
        return pcseq








        

