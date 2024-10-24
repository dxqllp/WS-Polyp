import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, generate
from models import build_model
from main import get_args_parser
from util.metrics import Metrics
from tqdm import tqdm
from util.metrics import evaluate
from val import valid
import os
from  torchvision.utils import   save_image


def main(args):
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_dict = model.state_dict()
    load_ckpt_path = os.path.join('./para/30permodel.pth')
    assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
    print('Loading checkpoint......')
    checkpoint = torch.load(load_ckpt_path)
    new_dict = {k : v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

    model.eval()
    dataset_val = build_dataset(image_set='test', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,                                
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    metrics_result = valid(model, data_loader_val, len(dataset_val),args,'test','30per')
    print("Valid Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f,'
              ' F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
              % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                 metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
                 metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
