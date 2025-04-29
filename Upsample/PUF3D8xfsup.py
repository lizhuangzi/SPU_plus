import os, sys

sys.path.append("../")
import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, Conv3d
from knn_cuda import KNN
from pointnet2.pointnet2_utils import gather_operation, grouping_operation
import torch.nn.functional as F
from arch.PixelShuffle1D import SpatialShuffle,SpatialShuffle_firstorder
import numpy as  np

class get_edge_feature(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """

    def __init__(self, k=16):
        super(get_edge_feature, self).__init__()
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.k = k

    def forward(self, point_cloud,input):
        dist, idx = self.KNN(point_cloud, point_cloud)
        '''
        idx is batch_size,k,n_points
        point_cloud is batch_size,n_dims,n_points
        point_cloud_neightbors is batch_size,n_dims,k,n_points
        '''
        idx = idx[:, 1:, :]
        point_cloud_neighbors = grouping_operation(point_cloud, idx.contiguous().int())
        point_cloud_central = input.unsqueeze(2).repeat(1, 1, self.k, 1,1)
        point_cloud_neighbors = point_cloud_neighbors.unsqueeze(3).repeat(1, 1, 1, 3,1)
        # print(point_cloud_central.shape,point_cloud_neighbors.shape)
        edge_feature = torch.cat([point_cloud_central, point_cloud_neighbors - point_cloud_central], dim=1)

        return edge_feature, idx

        return dist, idx

class get_edge_featureori(nn.Module):
    """construct edge feature for each point
    Args:
        tensor: input a point cloud tensor,batch_size,num_dims,num_points
        k: int
    Returns:
        edge features: (batch_size,num_dims,num_points,k)
    """

    def __init__(self, k=16):
        super(get_edge_featureori, self).__init__()
        self.KNN = KNN(k=k + 1, transpose_mode=False)
        self.k = k

    def forward(self, point_cloud):
        dist, idx = self.KNN(point_cloud, point_cloud)
        '''
        idx is batch_size,k,n_points
        point_cloud is batch_size,n_dims,n_points
        point_cloud_neightbors is batch_size,n_dims,k,n_points
        '''
        idx = idx[:, 1:, :]
        point_cloud_neighbors = grouping_operation(point_cloud, idx.contiguous().int())
        point_cloud_central = point_cloud.unsqueeze(2).repeat(1, 1, self.k, 1)
        # print(point_cloud_central.shape,point_cloud_neighbors.shape)
        edge_feature = torch.cat([point_cloud_central, point_cloud_neighbors - point_cloud_central], dim=1)

        return edge_feature, idx

        return dist, idx

class resconvori(nn.Module):
    def __init__(self, in_channels=64,k=16):
        super(resconvori, self).__init__()
        self.edge_feature_model = get_edge_featureori(k=k)
        self.in_channels = in_channels*2
        self.expansionrate = 2
        '''
        input to conv1 is batch_size,2xn_dims,k,n_points
        '''
        self.conv1 = nn.Sequential(
            Conv2d(in_channels=self.in_channels, out_channels=self.in_channels*self.expansionrate, kernel_size=[1, 1], padding=[0, 0]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            Conv2d(in_channels=self.in_channels*self.expansionrate, out_channels=self.in_channels*self.expansionrate, kernel_size=[1, 1], padding=[0, 0]),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            Conv2d(in_channels=self.in_channels*self.expansionrate, out_channels=self.in_channels//2, kernel_size=[1, 1],
                   padding=[0, 0]),
        )

    def forward(self, input):
        '''
        y should be batch_size,in_channel,k,n_points
        '''
        y, idx = self.edge_feature_model(input)
        res = self.conv3(self.conv2(self.conv1(y)))
        res = torch.max(res, dim=2)[0]  # pool the k channel
        out = res+input

        return out


class resconv(nn.Module):
    def __init__(self, in_channels=64,k=16):
        super(resconv, self).__init__()
        self.edge_feature_model = get_edge_feature(k=k)
        self.k = k
        self.in_channels = in_channels*2
        self.expansionrate = 2
        '''
        input to conv1 is batch_size,2xn_dims,k,n_points
        '''

        self.conv1 = nn.Sequential(
            Conv3d(in_channels=self.in_channels, out_channels=self.in_channels*self.expansionrate, kernel_size=[1, 1,1], padding=[0, 0,0]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            Conv3d(in_channels=self.in_channels*self.expansionrate, out_channels=self.in_channels*self.expansionrate, kernel_size=[1, 1,1], padding=[0, 0,0]),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            Conv3d(in_channels=self.in_channels*self.expansionrate, out_channels=self.in_channels//2, kernel_size=[1, 1,1],
                   padding=[0, 0,0]),
        )

    def forward(self, input):
        '''
        y should be batch_size,in_channel,k,n_points
        '''
        B,C,D,N = input.size()
        meaninput = torch.mean(input,dim=2,keepdim=False)
        edgefeatrue, idx1 = self.edge_feature_model(meaninput,input)


        # catresult = edgefeatrue.transpose(1,3)
        # outtrans = self.localcotrans1(catresult)
        # outtrans = outtrans.transpose(1,3)
        res = self.conv3(self.conv2(self.conv1(edgefeatrue)))
        # res = res.permute(0,3,1,2,4)
        # res = self.localcotrans2(res)
        # res = res.permute(0,2,3,1,4)
        res = torch.max(res, dim=2)[0]  # pool the k channel
        out = res+input

        return out


class feature_extraction(nn.Module):
    def __init__(self,inchannel=128):
        super(feature_extraction, self).__init__()
        self.reschannel = inchannel

        self.conv1 = nn.Sequential(
            Conv1d(in_channels=3, out_channels=self.reschannel, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            Conv1d(in_channels=self.reschannel, out_channels=self.reschannel*3, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.res1conv = resconvori(in_channels=self.reschannel)
        self.res1conv2 = resconvori(in_channels=self.reschannel)

        self.res1conv3 = resconv(in_channels=self.reschannel)
        self.res1conv4 = resconv(in_channels=self.reschannel)


    def forward(self, x):
        B, D, N = x.size()
        #x = x.unsqueeze(dim=1)
        f0 = self.conv1(x)  # b,64,3,n
        #f0 = f0.unsqueeze(dim=1).view(B, self.reschannel,3, N)

        #z_a = f0.view(B,D//3,3,N)


        f1 = self.res1conv(f0)
        f2 = self.res1conv2(f1)
        f2 = self.conv2(f2)
        f2 = f2.unsqueeze(dim=1)
        f2 = f2.view(B,self.reschannel,3,N)
        f3 = self.res1conv3(f2)
        f4 = self.res1conv4(f3)

        #z_b = x.unsqueeze(1).repeat(1, D // 3, 1, 1)
        # z_a_norm = (z_a-z_a.mean(dim=0))/z_a.std(dim=0)
        # z_b_norm = (z_b-z_b.mean(dim=0))/z_b.std(dim=0)
        #c = torch.matmul(f4,z_b.transpose(2,3))
        #c_diff = (c-torch.eye(3).cuda()).pow(2)

        return f4





class SEPS(nn.Module):
    def __init__(self,inchannel,outchannel = 256,up_ratio=4):
        super(SEPS, self).__init__()
        self.up_ratio = up_ratio
        self.conv0=nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1),
            nn.ReLU()
        )
        # self.conv1=nn.Sequential(
        #     nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1),
        #     nn.ReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=outchannel*2,out_channels=outchannel,kernel_size=1),
        #     nn.ReLU()
        # )

        self.convspe = Conv2d(in_channels=outchannel,out_channels=2*outchannel,kernel_size=1)
        self.convspe2 = Conv2d(in_channels=outchannel, out_channels=2 * outchannel,kernel_size=1)
        self.convspe3 = Conv2d(in_channels=outchannel, out_channels=2 * outchannel, kernel_size=1)
        self.upshuffle = SpatialShuffle_firstorder(self.up_ratio // 2, inchannel=2 * outchannel)
        # self.upshuffle2 = SpatialShuffle_plus(self.up_ratio // 2, inchannel=2 * outchannel)
        # self.upshuffle3 = SpatialShuffle_plus(self.up_ratio // 2, inchannel=2 * outchannel)



    def forward(self,x, f2=None,stage=0):
        B, C, D,N = x.shape

        x = self.conv0(x)

        x = self.convspe(x)
        x = self.upshuffle(x)
        x = self.convspe2(x)
        x = self.upshuffle(x)
        x = self.convspe3(x)
        x = self.upshuffle(x)
        return x

class Generator(nn.Module):
    def __init__(self, reschannel=128):
        super(Generator, self).__init__()
        self.feature_extractor = feature_extraction(inchannel=reschannel)

        self.upsampling = SEPS(inchannel=reschannel)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        )
        self.upratio = 4


    def forward(self, input):

        features  = self.feature_extractor(input)
        B,C,N,D = features.size()
        upf = self.upsampling(features)
        B, C, D,N = upf.size()
        coord = self.conv1(upf)
        coord = self.conv2(coord)
        coord = coord.squeeze(dim=1)
        return coord + nn.functional.interpolate(input=input,scale_factor=8,mode='nearest')

    def halfforwad(self, input, previous_features, globalfeature):

        previous_features = self.feature_extractor(input)
        upf = self.upsampling(previous_features, globalfeature, stage=1)
        coord = self.conv1(upf)
        coord = self.conv2(coord)
        return coord + F.upsample_nearest(input, scale_factor=4)

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        "up_ratio": 4,
        "patch_num_point": 100
    }
    generator = Generator(reschannel=48).cuda()
    print('# model parameters:', sum(param.numel() for param in generator.parameters()))
    point_cloud = torch.rand(4, 3, 100).cuda()
    output = generator(point_cloud)
    print(output.shape)
