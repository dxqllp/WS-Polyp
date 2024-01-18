# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .kvasir_SEG import kvasir_SEG

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'polyp' and image_set =='train':
        return kvasir_SEG('/home/E22201060/Data/'+args.train_per,'full',mode='train')
    if args.dataset_file == 'polyp' and image_set =='valid':
        return kvasir_SEG('/home/E22201060/Data/'+args.train_per,'weak',mode='valid')
    if args.dataset_file == 'polyp' and image_set =='test':
        return kvasir_SEG('/home/E22201060/Data/','test',mode='valid')

    # if args.dataset_file == 'coco':
    #     return build_coco(image_set, args)
    # if args.dataset_file == 'coco_panoptic':
    #     # to avoid making panopticapi required for coco
    #     from .coco_panoptic import build as build_coco_panoptic
    #     return build_coco_panoptic(image_set, args)
    # raise ValueError(f'dataset {args.dataset_file} not supported')
