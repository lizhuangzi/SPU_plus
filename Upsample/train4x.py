import os, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-e', type=str, required=False, help='experiment name')
parser.add_argument('--debug', action='store_true', help='specify debug mode')
parser.add_argument('--use_gan', action='store_true')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--gpu', type=str, default='3')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sys.path.append('../')
import torch
from Upsample.PUF3D4xfsp import Generator
from data.ModelNet40SPU import ModelNet40spu
import time
from option.train_option import get_train_options
from torch.utils import data
from torch.optim import Adam,lr_scheduler
from loss.loss import Loss
import torch.nn as nn
from ClsNetwork.PointNetcls2 import PoiintNet
from tqdm import tqdm
import pandas as pd
import numpy as np
from Clsutils.eval import accuracy
from Clsutils.misc import AverageMeter
import random
import sklearn.metrics as metrics


def xavier_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



def train(args):
    start_t = time.time()
    params = get_train_options()
    params["exp_name"] = 'Test1'
    params["patch_num_point"] = 1024
    params["batch_size"] = args.batch_size
    params['use_gan'] = args.use_gan

    if args.debug:
        params["nepoch"] = 2
        params["model_save_interval"] = 3
        params['model_vis_interval'] = 3

    log_dir = os.path.join(params["model_save_dir"], params["exp_name"])
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)

    ########### load dataset
    num_workers = 4
    trainloader = ModelNet40spu(data_dir=params["dataset_dir"], partition='train', num_points=params["num_points"],upscalefactor=4)
    train_data_loader = data.DataLoader(dataset=trainloader, batch_size=params["batch_size"], shuffle=True,
                                        num_workers=num_workers, pin_memory=True, drop_last=True)

    eval_dst = ModelNet40spu(data_dir=params["dataset_dir"], partition='test', num_points=params["num_points"],upscalefactor=4)
    val_loader = data.DataLoader(eval_dst, batch_size=25,
                             shuffle=False, pin_memory=True, num_workers=num_workers)



    G_model = Generator(reschannel=56)
    G_model.cuda()

    cls_model = PoiintNet(k=params['n_class'],normal_channel=False)
    cls_model.load_state_dict(torch.load('../PretrainModel/'+'PointNetModelNet40.parm'))
    cls_model.cuda()
    cls_model.eval()

    #optimizer_G = Adam(itertools.chain(G_model.parameters(),downsmnet.parameters()), lr=params["lr_G"], betas=(0.9, 0.999))
    optimizer_G = Adam(G_model.parameters(), lr=params["lr_G"],betas=(0.9, 0.999))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_G,T_max=120)

    Loss_fn = Loss()
    ValEXPresult = {'EMD': [],'CD': [],'Top1acc': [],'Top5acc': [],'Clsave':[]}
    RunningExpResult = {'EMDloss': [],'Uniformloss': [],'Distilllloss': []}
    Bestacc = 0.0; globalepoch=0

    print("preparation time is %fs" % (time.time() - start_t))

    #begin epoch
    for e in range(120):
        globalepoch += 1
        ########## train
        G_model.train()
        traincount = 0
        tqdmtrain = tqdm(train_data_loader)

        for batch_id, (input_data, gt_data,gtlabel) in enumerate(tqdmtrain):
            optimizer_G.zero_grad()
            cls_model.zero_grad()

            gtlabel = gtlabel.squeeze().cuda()
            input_data = input_data[:, :, 0:3].permute(0, 2, 1).float().cuda()
            gt_data = gt_data[:, :, 0:3].permute(0, 2, 1).float().cuda()
            realClsResult, feature_gt, _ = cls_model(gt_data)


            output_point_cloud = G_model(input_data)
            generateClsResult,feature_sr,_ = cls_model(output_point_cloud)


            clsloss = Loss_fn.corss_entropy(generateClsResult,gtlabel)

            #repulsion_loss = Loss_fn.get_repulsion_loss(output_point_cloud.permute(0, 2, 1))
            feature_sim = Loss_fn.mse_loss(feature_sr,feature_gt)

            total_G_loss = feature_sim

            tqdmtrain.set_description(desc='epoch%d  total_G_loss:%f ' % (e, total_G_loss.item()))
            traincount+=1


            total_G_loss.backward()
            optimizer_G.step()

        scheduler.step()
        ####### validation

        top1 = AverageMeter()
        top5 = AverageMeter()
        cd_list = []
        emd_list = []
        test_true = []
        test_pred = []
        G_model.eval()
        tqdm_test = tqdm(val_loader)
        for itr, batch in enumerate(tqdm_test):

            points, gt,gtlabel = batch

            gtlabel = gtlabel.squeeze().cuda()
            points = points[..., :3].permute(0, 2, 1).float().cuda().contiguous()
            gt = gt[..., :3].float().cuda().contiguous()

            # classification

            preds = G_model(points)
            preds = preds.detach()
            generateClsResult, _,_ = cls_model(preds)
            # clean grident
            cls_model.zero_grad()
            G_model.zero_grad()

            prec1, prec5 = accuracy(generateClsResult,gtlabel,topk=(1,5))
            top1.update(prec1.cpu().numpy(),points.size(0))
            top5.update(prec5.cpu().numpy(), points.size(0))

            pred = generateClsResult.max(dim=1)[1]
            test_true.append(gtlabel.cpu().numpy())
            test_pred.append(pred.detach().cpu().numpy())

            preds = preds.permute(0, 2, 1).contiguous()
            emd = Loss_fn.get_emd_loss(preds, gt, 1.0)
            cd = Loss_fn.get_cd_loss(preds, gt, 1.0)
            emd_list.append(emd.item())
            cd_list.append(cd.item())


        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        meanemd = np.mean(emd_list)
        meancd = np.mean(cd_list)

        ValEXPresult['EMD'].append(meanemd)
        ValEXPresult['CD'].append(meancd)
        ValEXPresult['Top1acc'].append(top1.avg)
        ValEXPresult['Top5acc'].append(top5.avg)
        ValEXPresult['Clsave'].append(class_acc)

        if Bestacc < top1.avg:
            Bestacc = top1.avg
            torch.save(G_model.state_dict(), './savedModel/PUF3D_Pre4x_fsup.parm')

        out_path = './statistics/'
        data_frame = pd.DataFrame(
            data={'EMD': ValEXPresult['EMD'],'CD':ValEXPresult['CD'],'Top1acc':ValEXPresult['Top1acc'],'Top5acc':ValEXPresult['Top5acc'],'Clsave':ValEXPresult['Clsave']},
            index=range(1, globalepoch + 1))
        #8x end 85.3
        #end2 0.0038 84.5 80.69
        # end3 85.1

        # mid 4x 0.0028 88.16 84.35
        #pre4 x2 88.3
        data_frame.to_csv(out_path + 'PUF3D_Pre4x_fsup' + '.csv', index_label='Epoch')





if __name__ == "__main__":
    seed = 1234
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    train(args)