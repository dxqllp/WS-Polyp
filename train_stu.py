##以分割比例为30为例蒸馏学生模型
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import datasets
from util.metrics import evaluate
from util.comm import generate_model
from util.loss import DeepSupervisionLoss,  BceDiceLoss,DeepSupervisionLoss1
from util.metrics import Metrics

import argparse
from main import get_args_parser
from models import build_model
import os
import util.misc as utils
from stu_model import ACSNet
from stu_model1 import CCBANet
from datasets import build_dataset, get_coco_api_from_dataset
import torch.nn.functional as F
from  torchvision.utils import   save_image
def valid(model, valid_dataloader, total_batch,args):

    model.eval()
    device = torch.device(args.device)

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    with torch.no_grad():
        for i, (samples, points, targets,filename) in enumerate(valid_dataloader):

            samples = samples.to(device)
            points = [{k: v.to(device) for k, v in t.items()} for t in points]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            image, mask = samples.decompose()
            output = model(image)
            if not os.path.exists('/Share/home/10014/zhangxuejun_stu/KDfeature/acskva/' + args.train_per):
                os.makedirs('/Share/home/10014/zhangxuejun_stu/KDfeature/acskva/' + args.train_per)

            save_image(output[0], '/Share/home/10014/zhangxuejun_stu/KDfeature/acskva/' + args.train_per + '/{}.png'.format(filename[0].split('/')[-1]))
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, targets)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, 
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, 
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean
                        )

    metrics_result = metrics.mean(total_batch)

    return metrics_result


def train(args):
    #硬标签列表
    hardlist = os.listdir('/Share/home/10014/zhangxuejun_stu/Data/{}/full/images'.format(args.train_per))

    #生成教师模型
    device = torch.device(args.device)
    teacher_model, criterion, postprocessors = build_model(args)
    teacher_model.to(device)

    model_dict = teacher_model.state_dict()
    load_ckpt_path = os.path.join('/Share/home/10014/zhangxuejun_stu/KDfeature/para/{}model.pth'.format(args.train_per))
    assert os.path.isfile(load_ckpt_path), 'No checkpoint found.'
    print('Loading checkpoint......')
    checkpoint = torch.load(load_ckpt_path)
    new_dict = {k : v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    teacher_model.load_state_dict(model_dict)
    teacher_model.eval()
    

    #生成学生模型
    stu_model = ACSNet(1)
    stu_model.to(device)

    #生成映射器
    project1 = torch.nn.Sequential(
        torch.nn.Conv2d(512,128,3,1,1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(True)
    ).to(device)
    project2 = torch.nn.Sequential(
        torch.nn.Conv2d(256,128,3,1,1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(True)
    ).to(device)
    project3 = torch.nn.Sequential(
        torch.nn.Conv2d(128,128,3,1,1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(True)
    ).to(device)
    project4 = torch.nn.Sequential(
        torch.nn.Conv2d(64,128,3,1,1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(True)
    ).to(device)
    project5 = torch.nn.Sequential(
        torch.nn.Conv2d(64,128,3,1,1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(True)
    ).to(device)

    #导入学生模型的训练集
    dataset_train = build_dataset(image_set='KD', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,                               
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers)


    dataset_val = build_dataset(image_set='test', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val) 
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,                                
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # load optimizer and scheduler
    optimizer = torch.optim.SGD(stu_model.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.weight_decay)
    optimizer.add_param_group({'params':project5.parameters()})
    optimizer.add_param_group({'params':project4.parameters()})
    optimizer.add_param_group({'params':project3.parameters()})
    optimizer.add_param_group({'params':project2.parameters()})
    optimizer.add_param_group({'params':project1.parameters()})


    lr_lambda = lambda epoch: 1.0 - pow((epoch / args.nEpoch), args.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # train
    print('Start training')
    print('---------------------------------\n')

    for epoch in range(args.nEpoch):
        print('------ Epoch', epoch + 1)
        stu_model.train()

        
        for index,(samples, points, targets,fname) in enumerate(data_loader_train):
            samples = samples.to(device)
            points = [{k: v.to(device) for k, v in t.items()} for t in points]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                outputs_logits = teacher_model(samples, points)
                pred0 = outputs_logits['pred_masks'][0]
            
            hard_index = []
            for k,v in enumerate(fname):
                if v+'.png'  in hardlist:
                    hard_index.append(k)

            image, mask = samples.decompose()
            output = stu_model(image)


            ##KD损失
            loss_soft = DeepSupervisionLoss1(output,torch.sigmoid(pred0))

            ##KD特征损失
            feature_stu5,feature_stu4,feature_stu3,feature_stu2,feature_stu1 = output[5::]
            feature_tea5,feature_tea4,feature_tea3,feature_tea2,feature_tea1 = outputs_logits['pred_masks'][1::]

            # feature_loss_soft5 = torch.nn.MSELoss()(F.normalize(project1(feature_stu5).flatten(2),2,2,1e-24),F.normalize(feature_tea5.flatten(2),2,2,1e-24))
            # feature_loss_soft4 = torch.nn.MSELoss()(F.normalize(project2(feature_stu4).flatten(2),2,2,1e-24),F.normalize(feature_tea4.flatten(2),2,2,1e-24))
            # feature_loss_soft3 = torch.nn.MSELoss()(F.normalize(project3(feature_stu3).flatten(2),2,2,1e-24),F.normalize(feature_tea3.flatten(2),2,2,1e-24))
            # feature_loss_soft2 = torch.nn.MSELoss()(F.normalize(project4(feature_stu2).flatten(2),2,2,1e-24),F.normalize(feature_tea2.flatten(2),2,2,1e-24))
            # feature_loss_soft1 = torch.nn.MSELoss()(F.normalize(project5(feature_stu1).flatten(2),2,2,1e-24),F.normalize(feature_tea1.flatten(2),2,2,1e-24))

            # feature_loss_soft5 = torch.abs(F.normalize(project1(feature_stu5).flatten(2),2,2,1e-24) - F.normalize(feature_tea5.flatten(2),2,2,1e-24)).flatten(1).mean()
            # feature_loss_soft4 = torch.abs(F.normalize(project2(feature_stu4).flatten(2),2,2,1e-24) - F.normalize(feature_tea4.flatten(2),2,2,1e-24)).flatten(1).mean()
            # feature_loss_soft3 = torch.abs(F.normalize(project3(feature_stu3).flatten(2),2,2,1e-24) - F.normalize(feature_tea3.flatten(2),2,2,1e-24)).flatten(1).mean()
            # feature_loss_soft2 = torch.abs(F.normalize(project4(feature_stu2).flatten(2),2,2,1e-24) - F.normalize(feature_tea2.flatten(2),2,2,1e-24)).flatten(1).mean()
            # feature_loss_soft1 = torch.abs(F.normalize(project5(feature_stu1).flatten(2),2,2,1e-24) - F.normalize(feature_tea1.flatten(2),2,2,1e-24)).flatten(1).mean()


            # tea_norm5 = F.softmax(feature_tea5.flatten(1),1)
            # stu_norm5 = F.softmax(project1(feature_stu5).flatten(1),1)
            # tea_norm4 = F.softmax(feature_tea4.flatten(1),1)
            # stu_norm4 = F.softmax(project2(feature_stu4).flatten(1),1)
            # tea_norm3 = F.softmax(feature_tea3.flatten(1),1)
            # stu_norm3 = F.softmax(project3(feature_stu3).flatten(1),1)
            # tea_norm2 = F.softmax(feature_tea2.flatten(1),1)
            # stu_norm2 = F.softmax(project4(feature_stu2).flatten(1),1)
            # tea_norm1 = F.softmax(feature_tea1.flatten(1),1)
            # stu_norm1 = F.softmax(project5(feature_stu1).flatten(1),1)
###sigmoid
            # tea_norm5 = F.sigmoid(feature_tea5.flatten(1))
            # stu_norm5 = F.sigmoid(project1(feature_stu5).flatten(1))
            # tea_norm4 = F.sigmoid(feature_tea4.flatten(1))
            # stu_norm4 = F.sigmoid(project2(feature_stu4).flatten(1))
            # tea_norm3 = F.sigmoid(feature_tea3.flatten(1))
            # stu_norm3 = F.sigmoid(project3(feature_stu3).flatten(1))
            # tea_norm2 = F.sigmoid(feature_tea2.flatten(1))
            # stu_norm2 = F.sigmoid(project4(feature_stu2).flatten(1))
            # tea_norm1 = F.sigmoid(feature_tea1.flatten(1))
            # stu_norm1 = F.sigmoid(project5(feature_stu1).flatten(1))


            # feature_loss_soft5 = torch.nn.KLDivLoss(reduction='batchmean')(stu_norm5.log(),tea_norm5)
            # feature_loss_soft4 = torch.nn.KLDivLoss(reduction='batchmean')(stu_norm4.log(),tea_norm4)
            # feature_loss_soft3 = torch.nn.KLDivLoss(reduction='batchmean')(stu_norm3.log(),tea_norm3)
            # feature_loss_soft2 = torch.nn.KLDivLoss(reduction='batchmean')(stu_norm2.log(),tea_norm2)
            # feature_loss_soft1 = torch.nn.KLDivLoss(reduction='batchmean')(stu_norm1.log(),tea_norm1)
            we352=torch.sigmoid(pred0)
            we176 = F.interpolate(we352, scale_factor=0.5, mode='bilinear', align_corners=True)
            we88 = F.interpolate(we176, scale_factor=0.5, mode='bilinear', align_corners=True)
            we44 = F.interpolate(we88, scale_factor=0.5, mode='bilinear', align_corners=True)
            we22 =F.interpolate(we44, scale_factor=0.5, mode='bilinear', align_corners=True)
            feature_loss_soft5 = torch.nn.MSELoss()((we22.flatten(2))*(F.normalize(project1(feature_stu5).flatten(2),2,2,1e-24)),(we22.flatten(2))*(F.normalize(feature_tea5.flatten(2),2,2,1e-24)))
            feature_loss_soft4 = torch.nn.MSELoss()((we44.flatten(2))*F.normalize(project2(feature_stu4).flatten(2),2,2,1e-24),(we44.flatten(2))*F.normalize(feature_tea4.flatten(2),2,2,1e-24))
            feature_loss_soft3 = torch.nn.MSELoss()((we88.flatten(2))*F.normalize(project3(feature_stu3).flatten(2),2,2,1e-24),(we88.flatten(2))*F.normalize(feature_tea3.flatten(2),2,2,1e-24))
            feature_loss_soft2 = torch.nn.MSELoss()((we176.flatten(2))*F.normalize(project4(feature_stu2).flatten(2),2,2,1e-24),(we176.flatten(2))*F.normalize(feature_tea2.flatten(2),2,2,1e-24))
            feature_loss_soft1 = torch.nn.MSELoss()((we352.flatten(2))*F.normalize(project5(feature_stu1).flatten(2),2,2,1e-24),(we352.flatten(2))*F.normalize(feature_tea1.flatten(2),2,2,1e-24))

            feature_loss_soft = feature_loss_soft5 + feature_loss_soft4+ feature_loss_soft3 + feature_loss_soft2+ feature_loss_soft1

            ##硬标签损失  ,硬标签损失只对全注释进行损失计算
            if len(hard_index)!=0:
                loss_hard = DeepSupervisionLoss(output, targets,hard_index)
            else:
                loss_hard = 0.

            ##总损失
            if loss_hard != 0:
                loss = 0.7*loss_hard + 0.3*loss_soft + feature_loss_soft
            if loss_hard == 0:
                loss = loss_soft + feature_loss_soft
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        scheduler.step()
        print("losshard:{}".format(loss_hard))
        print("losssoft:{}".format(loss_soft))
        print("lossfeature:{}".format(feature_loss_soft))



        metrics_result = valid(stu_model, data_loader_val, len(data_loader_val),args)
        print("Valid Result:")
        print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f,'
                ' F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
                % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                    metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
                    metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))

        if ((epoch + 1) % 5 == 0):
            if not os.path.exists('/Share/home/10014/zhangxuejun_stu/KDfeature/stu_para/' + args.train_per):
                os.makedirs('/Share/home/10014/zhangxuejun_stu/KDfeature/stu_para/' + args.train_per)
            torch.save(stu_model.state_dict(), '/Share/home/10014/zhangxuejun_stu/KDfeature/stu_para/' + args.train_per+ "/ck_{}.pth".format(epoch + 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    train(args)
    print('ok')
