import torch.nn as nn
import torch.nn.functional as F
import torch
from util.sobel import SobelComputer



"""BCE loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = diceloss + bceloss

        return loss

class BceDiceLoss_lapulasi(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss_lapulasi, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()


    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = bceloss +diceloss

        return loss


""" Deep Supervision Loss"""


def DeepSupervisionLoss_no(out, targets):
    conv_lapulasi = torch.nn.Conv2d(1,1,(3,3),stride=1,padding=1,padding_mode='reflect',bias=False)
    conv_lapulasi.weight.data = torch.Tensor([[[
            [-1,-1,-1],
            [-1,8,-1],
            [-1,-1,-1],
        ]]]).cuda()


    pred,pr0,pr1,pr2,pr3,edge0,edge1,bbox,x4,gc = out['pred_masks'][::]

    pred =torch.sigmoid(pred)
    pr0 =torch.sigmoid(pr0)
    pr1 =torch.sigmoid(pr1)
    pr2 =torch.sigmoid(pr2)
    pr3 =torch.sigmoid(pr3)

    edge0 =torch.sigmoid(edge0)
    edge1 =torch.sigmoid(edge1)

    criterion = BceDiceLoss()

   



##此处需要修改
    masks = targets[0]['mask']
    for bs in range(1,len(targets)):
        mask = targets[bs]['mask']
        if len(masks.shape)==4:
            mask= torch.unsqueeze(mask,0)
            masks = torch.cat([masks,mask],dim=0)
            continue
        masks = torch.stack([masks,mask],dim=0)  
    if masks.shape[0] ==1:
        masks = masks.unsqueeze(0)
    edge_mask =masks
    
    criterion_pred = BceDiceLoss()
    loss_pred = criterion_pred(pred, masks)

    masks = F.interpolate(masks, scale_factor=0.5, mode='bilinear', align_corners=True)
    criterion0 =BceDiceLoss()
    loss0 = criterion0(pr0, masks)


    masks = F.interpolate(masks, scale_factor=0.5, mode='bilinear', align_corners=True)
    criterion1 =BceDiceLoss()
    loss1 = criterion1(pr1, masks)

    masks = F.interpolate(masks, scale_factor=0.5, mode='bilinear', align_corners=True)
    criterion2 =BceDiceLoss()
    loss2 = criterion2(pr2, masks)



    masks = F.interpolate(masks, scale_factor=0.5, mode='bilinear', align_corners=True)
    criterion3 =BceDiceLoss()
    loss3 = criterion3(pr3, masks)


    sobel_compute = SobelComputer()
    label_sobel = sobel_compute(edge_mask)

    label_sobel = F.interpolate(label_sobel, scale_factor=0.5, mode='bilinear', align_corners=True)
    criterion_edge0 = BceDiceLoss()
    loss5 = criterion_edge0(edge0, label_sobel)

    label_sobel = F.interpolate(label_sobel, scale_factor=0.5, mode='bilinear', align_corners=True)
    criterion_edge1 = BceDiceLoss()
    loss6 = criterion_edge1(edge1, label_sobel)



    return loss0 +loss1 +loss2 +loss3 +loss_pred +loss5+ loss6
    # return loss_pred +loss5+ loss6+loss_refine1+loss_refine_edge



