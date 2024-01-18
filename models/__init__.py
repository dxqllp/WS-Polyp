# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .position_encoding import build_position_encoding




def build_model(args):
    return build(args)
