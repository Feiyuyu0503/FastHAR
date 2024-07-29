# -*- coding: utf-8 -*-
import os,glob
import h5py
import numpy as np
from torch.utils.data import Dataset

def load_dataset(partition,dir=''):
   
    all_data = []
    all_label = []
    all_frame_length = []

    for h5_name in glob.glob(os.path.join('dataset',dir,'*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['tag'][:].astype('int64')
        frame_length = f['frame_length'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_frame_length.append(frame_length)
        
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_frame_length = np.concatenate(all_frame_length, axis=0)
    return all_data, all_label,all_frame_length

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def translate_pointcloud(pointcloud,trans_x,trans_y,delta_z,rotation_angle,rand_num,xyz1,xyz2,shift_prob,scale_prob):
    if scale_prob > 0.5 :
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    else:
        translated_pointcloud = pointcloud.astype('float32')
    if rand_num > 0.5:
        translated_pointcloud = rotate_point_cloud(translated_pointcloud,rotation_angle)
    if shift_prob > 0.5:
        trans = np.array([trans_x/1.5,trans_y/1.5,delta_z])
        translated_pointcloud = np.add(translated_pointcloud,trans).astype('float32')
    return translated_pointcloud

def rotate_point_cloud(pointcloud, rotation_angle):
    pointcloud = np.array(pointcloud)
    num_points = pointcloud.shape[0]
    for i in range(num_points):
        point = pointcloud[i]
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_theta, sin_theta, 0],
                                    [-sin_theta, cos_theta, 0],
                                    [0, 0, 1]])
        rotated_point = np.dot(point.reshape(-1, 3), rotation_matrix)
        pointcloud[i] = rotated_point
    return pointcloud


class PointCloudDataset(Dataset):
    def __init__(self, num_points, partition='train',dir='',stride=1):
        self.data, self.label,self.frame_length = load_dataset(partition,dir)
        self.num_points = num_points
        self.partition = partition     
        self.stride = stride   

    def __getitem__(self, item):
        frame_num = self.data.shape[1]
        frame_length = self.frame_length[item]
        one_sample = np.zeros((frame_num, self.num_points, 3))
        trans_x = np.random.uniform(low=-1.5,high=3.5)
        trans_y = np.random.uniform(low=-1.5,high=3.5)
        delta_z = np.random.uniform(low=-0.20,high=0.10)
        xyz1 = np.random.uniform(low=3.0/3.25, high=3.25/3.0, size=[3])
        xyz2 = np.random.uniform(low=-0.1, high=0.1, size=[3])
        rotation_angle = np.random.uniform(low=-1/4.0,high=1/4.0)*np.pi*0.25
        rand_num = np.random.rand()
        shift_prob = np.random.rand()
        scale_prob = np.random.rand()
        seq = 0
        for i in range(0,frame_length,self.stride):
            pointcloud = self.data[item][i][:self.num_points]   # random sample
            if self.partition == 'train':
                pointcloud = translate_pointcloud(pointcloud,trans_x,trans_y,delta_z,rotation_angle,rand_num,xyz1,xyz2,shift_prob,scale_prob)
                np.random.shuffle(pointcloud)
                #pass
            pointcloud = pc_normalize(pointcloud)
            one_sample[seq] = pointcloud
            seq += 1
        label = self.label[item]
    
        frame_length = int(np.ceil(frame_length/self.stride))

        return one_sample, label,frame_length

    def __len__(self):
        return self.data.shape[0]