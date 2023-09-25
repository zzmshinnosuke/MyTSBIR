# -*- coding: utf-8 -*-

import argparse
import os

def get_parser_test(prog='tsbir'):
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
                        default=32,
                        help='the batch size')
    
    parser.add_argument('--resume',
                        type=str,
                        default="./runs/latest_checkpoint.pth",
                        help='model postion')
    
    
    return parser
