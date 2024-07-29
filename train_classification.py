# -*- coding: utf-8 -*-
from distutils.util import strtobool
import torch
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from classification_model import pcseq_classifier
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from utils.data import PointCloudDataset
from utils.util import *
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
import datetime
from tensorboard_logger import Logger
import time


def _init_():

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')

def train(args, io):
    model_log = '%s/%s' % ("log",args.exp_name)
    logger = Logger(logdir=model_log, flush_secs=2)
    train_loader = DataLoader(PointCloudDataset(partition='train', num_points=args.num_points,dir = args.dir,stride=args.dataset_stride), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(PointCloudDataset(partition='test', num_points=args.num_points,dir=args.dir,stride=args.dataset_stride), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if 'dgcnn' in args.encoder:
        model = pcseq_classifier(args).to(device)
    elif args.encoder == 'pointnet':
        model = pcseq_classifier(args).to(device)
    elif args.encoder == 'pointnet2':
        model = pcseq_classifier(args).to(device)
    elif args.encoder == 'pointmlp':
        model = pcseq_classifier(args).to(device)
    else:    
        raise Exception("Not implemented")

    io.cprint(str(model))
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    model = nn.DataParallel(model)

    criterion = cal_loss
    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data ,label ,frame_length in train_loader:
            if '3d' in args.encoder:
                frame_length = (frame_length/args.depth).type(torch.int64)
            data, label, frame_length = data.to(device).type(torch.float32), label.to(device).squeeze(), frame_length.squeeze()
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data,frame_length)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1] 
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred) # merge all batchs (still index of max value of embedding)
       
        logger.log_value('train_loss', train_loss / count, step=epoch)
        logger.log_value('train_acc', metrics.accuracy_score(train_true, train_pred), step=epoch)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label ,frame_length in test_loader:
            if '3d' in args.encoder:
                frame_length = (frame_length/args.depth).type(torch.int64)
            data, label, frame_length = data.to(device).type(torch.float32), label.to(device).squeeze(), frame_length.squeeze()
            batch_size = data.size()[0]
            logits = model(data,frame_length)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        logger.log_value('test_loss', test_loss / count, step=epoch)
        logger.log_value('test_acc', test_acc,step=epoch)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' \
                % ( epoch,
                    test_loss*1.0/count,
                    test_acc,
                    avg_per_class_acc)
        io.cprint(outstr)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_test_acc': best_test_acc,
                    'num_points':args.num_points,'emb_dims':args.emb_dims,'hidden_dims':args.hidden_dims,'k':args.k,
                    'encoder':args.encoder,'num_classes':args.num_classes,'dropout':args.dropout}
            
            torch.save(state, 'outputs/%s/models/model.pth' % args.exp_name)
        curr_state = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_test_acc': best_test_acc,
                    'num_points':args.num_points,'emb_dims':args.emb_dims,'hidden_dims':args.hidden_dims,'k':args.k,
                    'encoder':args.encoder,'num_classes':args.num_classes,'dropout':args.dropout}
        torch.save(curr_state, 'outputs/%s/models/curr_model.pth' % args.exp_name)
    
    io.cprint('best epcoh: %d，best test acc: %.6f' % (best_epoch,best_test_acc))
    

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Sequence Point Cloud Human Action Recognition')
    parser.add_argument('--exp_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--encoder', type=str, default='conv3d_sa_dgcnn', metavar='N',
                        choices=['conv2d_dgcnn','conv2d_sa_dgcnn','conv3d_dgcnn','conv3d_sa_dgcnn','pointnet', 'pointnet2','pointmlp'],
                        help='Encoder to use, [conv2d_dgcnn,conv2d_sa_dgcnn,conv3d_dgcnn,conv3d_sa_dgcnn,pointnet,pointnet2,pointmlp]')
    parser.add_argument('--dataset', type=str, default='pointcloud', metavar='N',
                        choices=['pointcloud'])
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=strtobool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=256,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=256, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=6, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--hidden_dims', type=int, default=128, metavar='N',
                    help='hidden_dims')
    parser.add_argument('--num_classes', type=int, default=7, metavar='N',
                    help='num_classes')      
    parser.add_argument('--dir', type=str, default='',
            help='dataset dir') 
    parser.add_argument('--gpu', type=int, default=0,
        help='gpu id to train') 
    parser.add_argument('--depth', type=int, default=2,
        help='conv3d kernel depth')
    parser.add_argument('--sort', type=str, default='morton',
        help='[morton,simple_morton,rnn,random]')
    parser.add_argument('--dataset_stride', type=int, default=1,
        help='dataset stride') 
    args = parser.parse_args()

 
    if args.exp_name == None:
        args.exp_name = str(datetime.date.today().strftime('%m%d')) + '_n' + str(args.num_points) + '_emb' + str(args.emb_dims) + '_hidden' + str(args.hidden_dims)  + '_k' + str(args.k)+ str(time.time())
    else:
        args.exp_name = args.exp_name +str(datetime.date.today().strftime('%m%d'))+ '_n' + str(args.num_points) + '_emb' + str(args.emb_dims) + '_hidden' + str(args.hidden_dims)  + '_k' + str(args.k)+str(time.time())
    _init_()
    io = IOStream('outputs/' + args.exp_name + '/run.log')
    start_time = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    io.cprint(start_time+' :training start!')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
        end_time = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        io.cprint(end_time+' :training end!')
        delta = datetime.datetime.strptime(end_time, '%Y-%m-%d  %H:%M:%S') - datetime.datetime.strptime(start_time, '%Y-%m-%d  %H:%M:%S')
        io.cprint('total time: %s' % str(delta))
