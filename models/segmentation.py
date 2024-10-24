# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from models.modules import  ASM,NonLocalBlock,SELayer
import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


def _expand(tensor, length: int):
    tensor = tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
    return tensor

class edge_Block(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=1):
        super(edge_Block, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel,3,stride=1,padding=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(2*in_channel, out_channel,3,stride=1,padding=1),  # conv2d  -》 bn -》relu
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel,3,stride=1,padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel,3,stride=1,padding=1),
        )
        self.se = SELayer(128)

    def forward(self, x,gc):
        x0 = self.branch0(x)

        x0 =  torch.cat((x0,gc),dim=1)
        x1 = self.branch1(x0)
        out =x1 +x


        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.gn3 = torch.nn.GroupNorm(8, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn3(x)
        x = self.relu(x)

        return x
class BasicConv2d1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.gn3 = torch.nn.GroupNorm(4, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn3(x)
        x = self.relu(x)

        return x

class RFB_modified1_4(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified1_4, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )



    def forward(self, x):
        x0 = self.branch0(x)
        return x0
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel,l):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),   # conv2d  -》 bn -》relu
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3*(5-l), dilation=3*(5-l))
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5*(5-l), dilation=5*(5-l))
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7*(5-l), dilation=7*(5-l))
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.non_local = NonLocalBlock(in_channel)
    def forward(self, x):
        # print("x的shape{}".format(x.shape))
        x0 = self.branch0(x)
        x0 = self.non_local(x0)
        # print("x0的shape{}".format(x0.shape))
        x1 = self.branch1(x)

        # print("x1的shape{}".format(x1.shape))
        x2 = self.branch2(x)


        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_ms1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_ms2 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5 = BasicConv2d(384, 128, 1)
        self.conv6 = nn.Conv2d(8, 8, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x1, x2, x3,bbox,gaosi):

        x1 = self.conv_upsample1(self.upsample(self.upsample(x1)))
        x2 = self.conv_upsample2(self.upsample(x2))
        x3_x1 = x3-x1
        x3_x1 = self.conv_ms1(x3_x1)

        x2_x1 = x2-x1
        x2_x1 = self.conv_ms2(x2_x1)

        x = x3_x1 + x2_x1 +x1
       
        x = self.conv4(x)
        gc_channel = x
        gaosi = self.pool(self.pool(self.pool(self.pool(self.pool(gaosi)))))

        x = self.pool(self.pool(x))
        x = torch.cat([x,bbox,gaosi],dim=1)
        x =self.conv5(x)
        gc_channel = self.pool(gc_channel)
        return x,gc_channel

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(DecoderBlock, self).__init__()
        self.conv1 = BasicConv2d(in_channels,in_channels // 2,kernel_size=kernel_size,
                                 stride=stride , padding=padding)
        self.conv2 =BasicConv2d(in_channels // 2,out_channels,kernel_size=kernel_size,
                                 stride=stride , padding=padding)
        self.gn3 = torch.nn.GroupNorm(2, 68)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.upsample(x)

        return x
    
class DecoderBlock1(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(DecoderBlock1, self).__init__()
        self.conv1 = BasicConv2d1(in_channels,in_channels // 2,kernel_size=kernel_size,
                                 stride=stride , padding=padding)
        self.conv2 =BasicConv2d1(in_channels // 2,out_channels,kernel_size=kernel_size,
                                 stride=stride , padding=padding)
        self.gn3 = torch.nn.GroupNorm(2, 68)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.upsample(x)

        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels //4,out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        ######edge


        
        self.bb = RFB_modified1_4(8, 128)

        self.rfb0_1 = RFB_modified1_4(64, 128)
        self.rfb1_1 = RFB_modified1_4(256, 128)

        self.rfb2_1 = RFB_modified1_4(512, 128)
        self.rgb2_gloab = RFB_modified(128,128,2)
        self.rfb3_1 = RFB_modified1_4(1024, 128)
        self.rgb3_gloab =RFB_modified(128,128,3)
        self.rfb4_1 = RFB_modified1_4(256, 128)
        self.rfb4_gloab = RFB_modified(128, 128,4)

                # Sideout
        self.sideout0 = SideoutBlock(128, 1)
        self.sideout1 = SideoutBlock(128, 1)
        self.sideout2 = SideoutBlock(128, 1)
        self.sideout3 = SideoutBlock(128, 1)
        self.sideout4 = SideoutBlock(128, 1)

        # Decoder
        self.decoder0 = DecoderBlock(in_channels=384, out_channels=128)
        self.decoder1 = DecoderBlock(in_channels=384, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=384, out_channels=128)
        self.decoder3 = DecoderBlock(in_channels=384, out_channels=128)
        self.decoder4 = DecoderBlock(in_channels=128, out_channels=128)

        #SEmoudle

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.agg1 = aggregation(128)
        self.asm0 =ASM(128,384)
        self.asm1 =ASM(128,384)
        self.asm2 =ASM(128,384)
        self.asm3 =ASM(128,384)

        self.edge_en1 = edge_Block(128,128)
        self.edge_en2_1 = edge_Block(128,128)
        
        self.sideout_edge1 = SideoutBlock(128, 1)
        self.sideout_edge2 = SideoutBlock(128, 1)
        self.pool = nn.MaxPool2d(2)

        self.se = SELayer(8,2)

        self.conv_se =torch.nn.Conv2d(8,128,1)
        self.conv_se_gn = torch.nn.GroupNorm(8,128)
        self.conv_se_relu = torch.nn.ReLU(True)

        self.conv_se_tem =torch.nn.Conv2d(8,8,1)
        self.conv_se_gn_tem = torch.nn.GroupNorm(2,8)
        self.conv_se_relu_tem = torch.nn.ReLU(True)

        self.conv_se_gaosi =torch.nn.Conv2d(1,128,1)
        self.conv_se_gn_gaosi = torch.nn.GroupNorm(8,128)
        self.conv_se_relu_gaosi = torch.nn.ReLU(True)


    def forward(self,gaosi_kernel:Tensor, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        tem =bbox_mask.flatten(0,1)
        # tem =self.conv_se_gn_tem(tem)
        # tem =self.conv_se_gn_tem(tem)
        # tem =self.conv_se_relu_tem(tem)

        bbox_mask = bbox_mask.flatten(0,1)
        bbox_mask = self.conv_se(bbox_mask)
        bbox_mask = self.conv_se_gn(bbox_mask)
        bbox_mask = self.conv_se_relu(bbox_mask)

        gaosi_kernel = gaosi_kernel.repeat(1,128,1,1)



        x0_rfb = self.rfb0_1(fpns[3])        # channel -> 128
        x1_rfb = self.rfb1_1(fpns[2])        # channel -> 128

        x2_rfb = self.rfb2_1(fpns[1])        # channel -> 128     
        x2_gloab = self.rgb2_gloab(x2_rfb)

        x3_rfb = self.rfb3_1(fpns[0])        # channel -> 128
        x3_gloab =self.rgb3_gloab(x3_rfb)

        x4_rfb = self.rfb4_1(x)        # channel -> 128
        x4_gloab = self.rfb4_gloab(x4_rfb)        # channel -> 128

        # gc = self.conv_se(self.se(bbox_mask))
        gc,gc3_channel = self.agg1(x4_gloab,x3_gloab,x2_gloab,bbox_mask,gaosi_kernel)
        gc3 = self.up(gc)  #gc3 22*22
        gc2 = self.up(gc3)#gc2 44*44
        gc1 = self.up(gc2)#gc1 88*88
        gc0 = self.up(gc1)  # 176*176

        x0_rfb_edge1 = self.edge_en1(x0_rfb,gc0)
        edge0 = self.sideout_edge1(x0_rfb_edge1)

        x1_rfb_edge1 = self.edge_en2_1(x1_rfb,gc1)
        edge1 = self.sideout_edge2(x1_rfb_edge1)

        d4 = self.decoder4(x4_rfb)
        pred4 = self.sideout3(d4)

        d3 = self.decoder3(torch.cat((d4,self.pool(self.pool(torch.sigmoid(edge1)))*x3_rfb+x3_rfb,gc3),dim=1))
        pred3 = self.sideout2(d3)

        d2 = self.decoder2(torch.cat((d3,self.pool(torch.sigmoid(edge1))*x2_rfb+x2_rfb,gc2),dim=1))
        pred2 = self.sideout1(d2)

        d1 =self.decoder1(torch.cat((d2,torch.sigmoid(edge1)*x1_rfb+x1_rfb,gc1),dim=1))
        pred1 =self.sideout0(d1)

        d0 = self.decoder0(torch.cat((d1,torch.sigmoid(edge0)*x0_rfb+x0_rfb,gc0),dim=1))
        pred0=self.sideout4(d0)




        return pred0,d4,d3,d2,d1,d0


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(2048, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        # if mask is not None:
        #     weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        # weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        # weights = self.dropout(weights)
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[None], to_tuple(size), mode="bilinear").squeeze(0)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds
