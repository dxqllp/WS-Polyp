import torch
from tqdm import tqdm
from util.metrics import Metrics,evaluate
from torchvision.utils import save_image
import torch.nn.functional as F
import os




def valid(model, valid_dataloader, total_batch,args,*image_set):

        model.eval()

        # Metrics_logger initialization
        metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                        'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

        with torch.no_grad():
            for i, (samples, points, targets,filename) in enumerate(valid_dataloader):

                samples = samples.to(args.device)
                points = [{k: v.to(args.device) for k, v in t.items()} for t in points]
                targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
                output = model(samples, points)
                pred,pr0,pr1,pr2,pr3 ,edge0,edge1,bbox,x4,gc0= output['pred_masks'][::]
                if len(image_set) !=0:
                    if not os.path.exists('./test_anyper/'+image_set[1]):
                        os.makedirs('./test_anyper/'+image_set[1])
                    save_image(torch.sigmoid(pred),'./test_anyper/'+image_set[1]+'/{}.png'.format(filename[0]))
                else:
                    if not os.path.exists('./result_vision/'+args.train_per+'/result'):
                        os.makedirs('./result_vision/'+args.train_per+'/result')
                    if not os.path.exists('./result_vision/'+args.train_per+'/edge'):
                        os.makedirs('./result_vision/'+args.train_per+'/edge')
                    save_image(torch.sigmoid(pred),'./result_vision/'+args.train_per+'/result'+'/{}.png'.format(filename[0]))
                # save_image(torch.sigmoid(pred),'./test.png')
                    save_image(torch.sigmoid(edge0),'./result_vision/'+args.train_per+'/edge'+'/{}.png'.format(filename[0]))
                # save_image(torch.sigmoid(gc0),'/Share/home/10014/zhangxuejun2/Point_refine_gn/gc0/{}.png'.format(filename[0]))
                gc = F.interpolate(bbox,scale_factor=32, mode='bilinear', align_corners=True)
                
                # for i in range(8):
                #     save_image(gc[0][i],"./inter_bbox/{}{}.png".format(filename[0],i))


                _recall, _specificity, _precision, _F1, _F2, \
                _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output,targets)

                metrics.update(recall= _recall, specificity= _specificity, precision= _precision, 
                                F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, 
                                IoU_bg= _IoU_bg, IoU_mean= _IoU_mean
                            )

        metrics_result = metrics.mean(total_batch)
        return metrics_result
    
