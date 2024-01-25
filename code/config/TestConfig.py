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
    
    parser.add_argument('--resume',
                        type=str,
                        default="./runs/SFSD/version_1/checkpoints/best-SFSD.ckpt",
                        help='model postion')
    
    parser.add_argument('--Top_K',
                        type=int,
                        default=1,
                        help='recall top_k')
    
    parser.add_argument('--pre_model',
                        default=False,
                        help='the path of checkpoint. if not, it is false')
    
    parser.add_argument('--output_dim',
                        type=int,
                        default=40,
                        help='the max epoch number')
    
    parser.add_argument('--input_type',
                        default='TS',
                        choices=['T','S','TS'],
                        help='input Text or Sketch or Text+Sketch')
    
    
    return parser
