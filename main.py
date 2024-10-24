# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from  torchvision.utils import   save_image
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import  train_one_epoch, generate
from models import build_model
from util.metrics import Metrics
from tqdm import tqdm
from util.metrics import evaluate
from val import valid
import datetime


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--nEpoch', default=370, type=int)
    parser.add_argument('--lr_drop', default=24, type=int)
    parser.add_argument('--multi_step_lr', action='store_true')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    #SGD
    parser.add_argument('--mt', type=float, default=0.9)
    parser.add_argument('--power', type=float, default=0.9)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='relative-learned', type=str, choices=('sine', 'learned', 'relative-learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pos_emb_relative_dim', default=51, type=int, help="position embedding .. ")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    # parser.add_argument('--set_cost_class', default=1, type=float,
    #                     help="Class coefficient in the matching cost")
    # parser.add_argument('--set_cost_bbox', default=1, type=float,
    #                     help="L1 box coefficient in the matching cost")
    # parser.add_argument('--set_cost_giou', default=2, type=float,
    #                     help="giou box coefficient in the matching cost")
    # * Loss coefficients
    # parser.add_argument('--mask_loss_coef', default=1, type=float)
    # parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--iou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='polyp')
    parser.add_argument('--train_per', default='30per')
    parser.add_argument('--coco_path',default='./datasets/COCO' ,type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./ckpts/baseline',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--visual_num', default=5, type=int)
    parser.add_argument('--visual_order', default='score', type=str, help='order by score or query')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--partial_training_data', action='store_true', help='only 20% of the training data.')
    parser.add_argument('--percent_of_training_data', default=20, type=int, help='1,2,5,10,20 for 1%,2%,5%,10%,20% COCO data')
    parser.add_argument('--data_augment', action='store_true', help='need data augment when training ?  ')
    parser.add_argument('--strong_aug', action='store_true', help='colorjitter + grayscale + gaussianblur')

    parser.add_argument('--without_crop', action='store_true', help='without size crop .. ')
    parser.add_argument('--generate_pseudo_bbox', action='store_true', help='xxxxxx')
    parser.add_argument('--generated_anno', default='undefined', type=str, help='generated pseudo_bbox annotation file name ..')
    parser.add_argument('--generated_point_idx', default=0, type=int, help='point idx , 0~9; -1 represent the center of bounding box')
    # parser.add_argument('--rlaunch -P1 --preemptible no --cpu 20 --gpu 8 --memory 200000 -- python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /data/Datasets/COCO --partial_training_data --output_dir ./ckpt-ps/no-aug-pos_dim_2-6x-logiou --pos_emb_relative_dim 2 --epochs 72 --lr_drop 48iou_loss_type', type=str, default='logiou', help='logiou, iou, giou')
    parser.add_argument('--iou_loss_type', type=str, default='giou', help='logiou, iou, giou')
    parser.add_argument('--warm_up', action='store_true', help='warm up, factor = 0.33333..')
    parser.add_argument('--dropout_points', action='store_true', help='keep 10 points most .. ')

    parser.add_argument('--no_label_encoder', action='store_true', help='no label encoder .. ')
    parser.add_argument('--no_pos_encoder', action='store_true', help='no pos encoder .. ')
    parser.add_argument('--self_attn_label_encoder', action='store_true', help='self attn label encoder .. ')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) #返回model的参数量
    para=model.parameters()

    print('number of params:', n_parameters)

    # param_dicts = [
    #     {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    # ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)
    # from torch.optim.lr_scheduler import LambdaLR
    # lr_lambda = lambda epoch: 1.0 - pow((epoch / args.epochs), args.power)
    # lr_scheduler = LambdaLR(optimizer, lr_lambda)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='valid', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if not args.eval and not args.generate_pseudo_bbox and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler.milestones = milestones
            args.start_epoch = checkpoint['epoch'] + 1

    if args.generate_pseudo_bbox:
        # args.generated_ann
        generate(model, data_loader_train, device, args.generated_anno, args.percent_of_training_data, args.coco_path)

        return



    # if args.eval:
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir)
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir, args.visual,
    #                                           args.visual_num, args.visual_order)
    #     # if args.output_dir:
    #     #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #     return

    print("Start training")
    start = datetime.datetime.now()
    print("开始时间为:",start)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        print(train_stats)
        lr_scheduler.step()
        torch.save(model.state_dict(),'./para/'+args.train_per+'model.pth')
    
        metrics_result = valid(model, data_loader_val, len(dataset_val),args)
        print('第{}epoch'.format(epoch))
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print("Valid Result:")
        print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f,'
              ' F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
              % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                 metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
                 metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))



    last = datetime.datetime.now()
    print("结束时间为:",last)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))





if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
