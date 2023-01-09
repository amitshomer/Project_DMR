from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import Models.resnet as resnet
from utils.utils_SA import samples_random, get_boundary_points_bn
from utils.utils_SA import prune as prune_func

import numpy as np

class Conv_layer(nn.Module):
    def __init__(self, feature_size_in, feature_size_out, kernel ):
        super(Conv_layer, self).__init__()
        self.conv = torch.nn.Conv1d(feature_size_in, feature_size_out, kernel)
        torch.nn.init.xavier_uniform(self.conv.weight)
        self.bn = torch.nn.BatchNorm1d(feature_size_out)
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))

        return x

class Error_Estimator(nn.Module):
    def __init__(self, feature_size = 1024):
        super(Error_Estimator, self).__init__()
        self.conv_layer1 = Conv_layer(feature_size, feature_size, 1)
        self.conv_layer2 = Conv_layer(feature_size, feature_size//2, 1)
        self.conv_layer3 = Conv_layer(feature_size//2, feature_size//4, 1)
        self.conv4 = torch.nn.Conv1d(feature_size//4, 1, 1)

        self.sig = nn.Sigmoid()
        # self.bn1 = torch.nn.BatchNorm1d(feature_size)
        # self.bn2 = torch.nn.BatchNorm1d(feature_size//2)
        # self.bn3 = torch.nn.BatchNorm1d(feature_size//4)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.sig(self.conv4(x))

        return x

class DeformNet(nn.Module):
    def __init__(self, feature_size = 1024):
        super(DeformNet, self).__init__()
        self.conv_layer1 = Conv_layer(feature_size, feature_size, 1)
        self.conv_layer2 = Conv_layer(feature_size, feature_size//2, 1)
        self.conv_layer3 = Conv_layer(feature_size//2, feature_size//4, 1)
        self.conv4 = torch.nn.Conv1d(feature_size//4, 3, 1)

        self.th = nn.Tanh()
        # self.bn1 = torch.nn.BatchNorm1d(feature_size)
        # self.bn2 = torch.nn.BatchNorm1d(feature_size//2)
        # self.bn3 = torch.nn.BatchNorm1d(feature_size//4)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.th(self.conv4(x))
        return x

        
class PointNet(nn.Module):
    def __init__(self, num_points=2500, feature_size=1024):
        super(PointNet, self).__init__()
        self.conv_layer1 = Conv_layer(3, 64, 1)
        self.conv_layer2 = Conv_layer(64, 128, 1)
        self.conv_layer3 = Conv_layer(128, 1024, 1)

        # self.bn1 = torch.nn.BatchNorm1d(64)
        # self.bn2 = torch.nn.BatchNorm1d(128)
        # self.bn3 = torch.nn.BatchNorm1d(1024)
        self.num_points = num_points
        self.linear = nn.Linear(1024, feature_size)
        self.bn4 = nn.BatchNorm1d(feature_size)

    def forward(self, x):
        # We must have BN here - without this we will have overfit
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = torch.max(x, 2)[0]
        x = x.contiguous()
        x = F.relu(self.bn4(self.linear(x)))
        return x


class Base_Img_to_Mesh(nn.Module):
    def __init__(self,
                 feature_size=1024,
                 num_points=2500):
        super(Base_Img_to_Mesh, self).__init__()
        self.feature_size = feature_size
        self.num_points = num_points

        self.point_cloud_encoder = PointNet(num_points=num_points, feature_size=feature_size)
        
        # TODO- take Resnet AsIs, maybe try VIT or somthing else
        self.img_endoer = self.encoder = resnet.resnet18(
            num_classes=feature_size)

        self.deformNet1 = DeformNet(feature_size=3 + feature_size)
        


    def forward(self, x, mode='point'):
        if mode == 'point':
            x = self.point_cloud_encoder(x)
        else:
            x = self.img_endoer(x)
        #TODO - check the rand grid
        rand_grid = torch.cuda.FloatTensor(x.size(0), 3, self.num_points)
        rand_grid.data.normal_(0, 1)
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True))\
            .expand(x.size(0), 3, self.num_points)
        y = x.unsqueeze(2).expand(x.size(0), x.size(1),
                                  rand_grid.size(2)).contiguous()
        y = torch.cat((rand_grid, y), 1).contiguous()
        outs = self.deformNet1(y)
        return outs.contiguous().transpose(2, 1).contiguous()

class Subnet1(nn.Module):
    def __init__(self,cuda, feature_size = 1024):
        super(Subnet1, self).__init__()
        self.cuda_device = cuda
        self.deformNet1 = DeformNet(feature_size=3 + feature_size)
        self.error_estimator = Error_Estimator(feature_size=3 + feature_size)

    def forward(self, args, img_featrue, points, faces_cuda, num_points, num_samples, prune,  tau=0.1):
        batch_size= img_featrue.size(0)
        if points.shape[2]==3:
            points = points.permute(0,2,1).contiguous()
        img_featrue1 = img_featrue.unsqueeze(2).expand(img_featrue.size(0), img_featrue.size(1), points.size(2)).contiguous()
        x = torch.cat((points, img_featrue1), 1).contiguous()
        # DeformNet 
        pointsRec = self.deformNet1(x)
        pointsRec = pointsRec.permute(0,2,1).contiguous()
        if prune:
            pointsRec_samples, index_sample = samples_random(faces_cuda, pointsRec.detach(), num_points, device= self.cuda_device)
        else:
            pointsRec_samples, index_sample = samples_random(faces_cuda.detach(), pointsRec, num_points, device= self.cuda_device)
        # Topology Modification
   
    
        random_choice = np.random.choice(num_points, num_samples, replace=False)
            
        if prune:
            img_featrue2 = img_featrue.unsqueeze(2).expand(batch_size, img_featrue.size(1), num_points).contiguous()
            pointsRec_samples_cat = torch.cat((pointsRec_samples.detach().transpose(1, 2), img_featrue2), 1).contiguous()
        else: 
            img_featrue2 = img_featrue.unsqueeze(2).expand(batch_size, img_featrue.size(1), num_samples).contiguous()
            pointsRec_samples_cat = torch.cat((pointsRec_samples.detach()[:,random_choice].transpose(1, 2), img_featrue2), 1).contiguous()
        
        out_error_estimator = self.error_estimator(pointsRec_samples_cat).squeeze() ## STOP here

        if prune:
            if faces_cuda.size(0) != batch_size:
                faces_cuda_bn = faces_cuda.unsqueeze(0).expand(batch_size, faces_cuda.size(0), faces_cuda.size(1))
            else:
                faces_cuda_bn = faces_cuda.to(self.cuda_device)
            t = args.tau
            if out_error_estimator.size(0) != batch_size:
                out_error_estimator = out_error_estimator.unsqueeze(0)
            #faces_cuda_bn_sa = utils_sa.prune_func(faces_cuda_bn.clone().detach(), out_error_estimator.clone().detach(), t, index_sample.clone(), 'max', device = self.cuda_device)
            faces_cuda_bn = prune_func(faces_cuda_bn.detach(), out_error_estimator.detach(), t, index_sample, 'max', device = self.cuda_device)
            #assert ((faces_cuda_bn != faces_cuda_bn_sa).sum() == 0 )
        else:
            faces_cuda_bn = None

        return pointsRec, pointsRec_samples, out_error_estimator, random_choice, faces_cuda_bn

class Refinement(nn.Module):
    def __init__(self, cuda, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size +3
        self.cuda = cuda
        super(Refinement, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 2, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
        self.th = nn.Tanh()

    def forward(self,points, img_featrue, faces_cuda_bn ):
        # Boundery 
        pointsRec2_boundary, selected_pair_all, selected_pair_all_len = get_boundary_points_bn(faces_cuda_bn, points, self.cuda)
        vec1 = (pointsRec2_boundary[:, :, 1] - pointsRec2_boundary[:, :, 0])
        vec2 = (pointsRec2_boundary[:, :, 2] - pointsRec2_boundary[:, :, 0])
        vec1 = vec1 / (torch.norm((vec1 + 1e-6), dim=2)).unsqueeze(2)
        vec2 = vec2 / (torch.norm((vec2 + 1e-6), dim=2)).unsqueeze(2)
        vec1 = vec1.transpose(2,1).detach()
        vec2 = vec2.transpose(2,1).detach()
        
        if pointsRec2_boundary.shape[1] != 0:
            batch_size= img_featrue.size(0)
            points = pointsRec2_boundary[:, :, 0]
            if points.shape[1] != 3:
                points= points.permute(0,2,1).contiguous()
            img_featrue1 = img_featrue.unsqueeze(2).expand(img_featrue.size(0), img_featrue.size(1), points.size(2)).contiguous()
            x = torch.cat((points, img_featrue1), 1).contiguous()
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.th(self.conv4(x))
            out = x[:, 0].unsqueeze(1) * vec1 + x[:, 1].unsqueeze(1) * vec2 + points
            out = out.permute(0,2,1).contiguous()
        else:
            out = pointsRec2_boundary[:, :, 0]
        
        displace_loss = out - pointsRec2_boundary[:, :, 0]
        displace_loss = torch.mean(torch.abs(displace_loss))
        
        return out, displace_loss, selected_pair_all.to(self.cuda), selected_pair_all_len