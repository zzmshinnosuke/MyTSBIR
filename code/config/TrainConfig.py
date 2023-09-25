# -*- coding: utf-8 -*-

import argparse
import os

def get_parser_train(prog='tsbir'):
    parser=argparse.ArgumentParser(prog)
    
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='no of gpus')
    # dataset
    parser.add_argument('--dataset',
                        required=True,
                        help='the dataset type')
    
    parser.add_argument('--dataset_root_path',
                        required=True,
                        help='the root path for dataset')
    
    parser.add_argument('--loader_num_workers',
                        type=int,
                        default=5,
                        help='the number of loader workers')
    
    # logger
    parser.add_argument('--logger_comment',
                        type=str,
                        default="tsbir_SFSD",
                        help='logger name')
    
    parser.add_argument('--logger_path',
                        type=str,
                        default="./runs/",
                        help='logger save path')
    
    # model
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='the batch size')
    
    parser.add_argument('--n_epoch',
                        type=int,
                        default=100,
                        help='the max epoch number')
    
    parser.add_argument('--input_dim',
                        type=int,
                        default=512,
                        help='the max epoch number')
    
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=256,
                        help='the max epoch number')
    
    parser.add_argument('--output_dim',
                        type=int,
                        default=40,
                        help='the max epoch number')
    
    #lr_scheduler:
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='optimizer learning_rate')
    
    parser.add_argument('--scheduler',
                        default='StepLR',
                        choices=['StepLR','ReduceLROnPlateau'],
                        help='lr_scheduler type')
    
    parser.add_argument('--learning_rate_decay_frequency',
                        type=int,
                        default=5,
                        help='lr_scheduler learning_rate_decay_frequency')
    
    parser.add_argument('--learning_rate_factor',
                        type=float,
                        default=0.5,
                        help='lr_scheduler learning_rate_factor')
    
    
    
    return parser
