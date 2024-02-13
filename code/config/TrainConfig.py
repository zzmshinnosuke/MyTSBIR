# -*- coding: utf-8 -*-

import argparse
import os

def get_parser_train(prog='tsbir'):
    parser=argparse.ArgumentParser(prog)
    
    parser.add_argument('--gpu_nums',
                        type = int,
                        default = 1,
                        help = 'the number of gpus')
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
    parser.add_argument('--model',
                        default="MultiGrainModel",
                        help = 'the model type')
    
    parser.add_argument('--model_config',
                        default="./model_pt/ViT-B-16.json",
                        help = 'the model config file')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='the batch size')
    
    parser.add_argument('--n_epoch',
                        type=int,
                        default=50,
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
    
    parser.add_argument('--input_type',
                        default='TS',
                        choices=['T','S','TS'],
                        help='input Text or Sketch or Text+Sketch')
    
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
    
     # resume：  
    parser.add_argument('--resume',
                        default=False,
                        help='model postion')

    parser.add_argument('--pre_model',
                        default=False,
                        # default="./model_pt/tsbir_model_final.pt",
                        help='the path of checkpoint. if not, it is false')
    
    parser.add_argument('--test_result_path',
                        help='the path for test result')
    return parser
