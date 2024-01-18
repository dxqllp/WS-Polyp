import os
import os.path as osp
from util.transform import *
from torch.utils.data import Dataset
from torchvision import transforms
import json
import numpy as np
from .center_map import center_map_gen,gaussian
from torchvision.utils import save_image

# KavSir-SEG Dataset
class kvasir_SEG(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(kvasir_SEG, self).__init__()
        data_path = osp.join(root, data2_dir)
        self.imglist = []
        self.gtlist = []
        self.points = []
        self.mode = mode
        datalist = os.listdir(osp.join(data_path, 'images'))
        for data in datalist:
            self.imglist.append(osp.join(data_path+'/images', data))
            self.gtlist.append(osp.join(data_path+'/masks', data))
            self.points.append(data_path+'/points/'+data.split('.')[0]+'.json')

        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize((352, 352)),
                   ToTensor(),

               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((352, 352)),
                   ToTensor(),
               ])
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imglist[index]
        gt_path = self.gtlist[index]
        content_path =self.points[index]

        file_name = img_path.split('/')[-1]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        with open(content_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        point_list = content['point']
        point_x = point_list[0]
        point_y = point_list[1]
        point = np.array([[point_x,point_y]])

        gaosi_kernel = torch.zeros((1,352,352))
        center_map = center_map_gen(gaosi_kernel, point_y, point_x, 0, 20, gaussian(20))




        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)
        # point = torch.tensor([[0,0]])
        point = torch.tensor(point)
        point = point/torch.tensor([352,352])
        

        point_supervion = {}
        target = {}



        point_supervion['point'] = point
        point_supervion['gaosi'] = center_map
        target['mask'] = data['label']

        # save_image(data['image'],'test_image1.png')


        return data['image'],point_supervion,target,file_name.split('.')[0]

    def __len__(self):
        return len(self.imglist)
