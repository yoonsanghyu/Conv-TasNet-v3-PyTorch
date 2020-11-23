# -*- coding: utf-8 -*-
"""
Created on 2018/12
Author: Kaituo XU

Edited by: yoonsanghyu 2020/11  

"""

import argparse

import os
import torch
import numpy as np


from data import AudioDataLoader, AudioDataset
from solver import Solver
from ConvTasNet_v3 import ConvTasNet



parser = argparse.ArgumentParser("Conv-TasNet\n" "Y. Luo")

# General config
# Task related
parser.add_argument('--tr-json', type=str, default='data/tr.json',
                    help='path to .json file for training')
parser.add_argument('--cv-json', type=str, default='data/cv.json',
                    help='path to .json file for validation')
parser.add_argument('--sample-rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
# Network architecture
parser.add_argument('--N', default=512, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=16, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=128, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--Sc', default=128, type=int,
                    help="Number of channels in skip-connection paths' 1 x 1-conv blocks")
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=3, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=2, type=int,
                    help='Maximum number of speakers')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
# Training config
parser.add_argument('--use-cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half-lr', dest='half_lr', default=1, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early-stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max-norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=1, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--drop', default=0, type=int,
                    help='drop files shorter than this')
parser.add_argument('--batch-size', default=3, type=int,
                    help='Batch size')
parser.add_argument('--num-workers', default=12, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save-folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
# logging
parser.add_argument('--print-freq', default=1000, type=int,
                    help='Frequency of printing training infomation')



def main(args):
    # Construct Solver
    
    # data
    tr_dataset = AudioDataset(args.tr_json, sample_rate=args.sample_rate, segment=args.segment, drop=args.drop)
    cv_dataset = AudioDataset(args.cv_json, sample_rate=args.sample_rate, drop=0, segment=-1)  # -1 -> use full audio
    tr_loader = AudioDataLoader(tr_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1, num_workers=0)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    
    # model
    # N=512, L=32, B=128, Sc=128, H=512, X=8, R=3, P=3, C=2
    model = ConvTasNet(args.N, args.L, args.B, args.Sc, args.H, args.X, args.R, args.P, args.C)

    print(model)
    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"]='5,6,7'
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
