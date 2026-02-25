import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel

from ara.utils.config import Config
from ara.engine.runner import Runner


def setup_parser():
    parser = argparse.ArgumentParser(description='Angle-Aware Rectangle Anchors (ARA) for Lane Detection')
    
    parser.add_argument('config', type=str, help='Path to the model configuration file')
    
    parser.add_argument('--work_dirs', type=str, default=None, 
                        help='Directory to save training logs and model checkpoints')
    parser.add_argument('--load_from', type=str, default=None, 
                        help='Path to pre-trained weights to load')
    parser.add_argument('--resume_from', type=str, default=None, 
                        help='Path to a checkpoint file to resume training from')
    parser.add_argument('--finetune_from', type=str, default=None, 
                        help='Path to a checkpoint file for fine-tuning')
    
    parser.add_argument('--view', action='store_true', help='Enable visualization mode')
    parser.add_argument('--validate', action='store_true', 
                        help='Run validation on the specified checkpoint')
    parser.add_argument('--test', action='store_true', 
                        help='Run testing on the target dataset')
    
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], 
                        help='List of GPU IDs to use for training/testing')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed to ensure reproducibility')
    
    return parser.parse_args()


def main():
    opts = setup_parser()
    
    gpu_ids = [str(gpu) for gpu in opts.gpus]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_ids)
    cudnn.benchmark = True  

    cfg = Config.fromfile(opts.config)
    cfg.gpus = len(opts.gpus)
    cfg.load_from = opts.load_from
    cfg.resume_from = opts.resume_from
    cfg.finetune_from = opts.finetune_from
    cfg.view = opts.view
    cfg.seed = opts.seed

    if opts.work_dirs is not None:
        cfg.work_dirs = opts.work_dirs

    ara_runner = Runner(cfg)

    if opts.test:
        print("=> Starting ARA testing mode...")
        ara_runner.test()
    elif opts.validate:
        print("=> Starting ARA validation mode...")
        ara_runner.validate()
    else:
        print("=> Starting ARA training mode...")
        ara_runner.train()


if __name__ == '__main__':
    main()