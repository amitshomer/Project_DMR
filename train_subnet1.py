## Trainer Partially based on the offical Repo of the paper - prints, general method, logger and so. 

import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as  dist_chamfer_3D
from Models.Main_models import Base_Img_to_Mesh as Base_network
from Models.Main_models import Subnet1 
from utils.utils import weights_init, AverageValueMeter, get_edges, create_round_spehere, load_weights
from utils.loss import get_edge_loss_stage1, smoothness_loss_parameters, mse_loss, get_edge_loss, get_smoothness_loss_stage1, get_normal_loss 
import argparse
from utils.dataset import ShapeNet
import os
import json
import torch
import torch.optim as optim
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help=' batch size')
parser.add_argument('--workers', type=int, default=8,  help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=420, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=10000, help='number of points for GT')
parser.add_argument('--num_samples',type=int,default=2500, help='number of samples for error estimation')
parser.add_argument('--dir_name', type=str, default="subnet1", help='')
parser.add_argument('--lr',type=float,default=1e-3, help='initial learning rate')
parser.add_argument('--num_vertices', type=int, default=2562, help='number of vertices of the initial sphere')
parser.add_argument('--model_path', type=str, default='./log/base_weights_run/network.pth', help='model path from the pretrained model')
parser.add_argument('--device', type=int, default=0, help='GPU device')

args = parser.parse_args()
print(args)
cuda = torch.device('cuda:{}'.format(args.device))
## Logger part ##
dir_name = os.path.join('./log', args.dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')

with open(logname, 'a') as f:  # open and append
    f.write(str("subnet1") + '\n')

## data loader part take from the offical repo
dataset = ShapeNet(npoints=args.num_points, SVR=True, normal=True, train=True, class_choice='chair')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers))
dataset_val = ShapeNet(npoints=args.num_points, SVR=True, normal=True, train=False, class_choice='chair')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                              shuffle=False, num_workers=int(args.workers))
len_dataset = len(dataset)

# Create the round mesh spehre 
edge_cuda, vertices_sphere, faces_cuda, faces =  create_round_spehere(args.num_vertices, cuda = 'cuda:0')
parameters = smoothness_loss_parameters(faces)

# Base model - load with weights
encoder = Base_network()
encoder = load_weights(encoder, args.model_path )
encoder = encoder.img_endoer

# Subnet1
subnet1 = Subnet1(cuda= cuda)
subnet1 = load_weights(subnet1, args.model_path )

# optimizer load
optimizer = optim.Adam([
    {'params': encoder.parameters()},
    {'params': subnet1.parameters()}
], lr=args.lr)


### Averge and prints take from the origianl repo
train_CD_loss = AverageValueMeter()
val_CD_loss = AverageValueMeter()
train_l2_loss = AverageValueMeter()
val_l2_loss = AverageValueMeter()
train_CDs_loss = AverageValueMeter()
val_CDs_loss = AverageValueMeter()
train_CD_curve = []
val_CD_curve = []
train_l2_curve = []
val_l2_curve = []
train_CDs_curve = []
val_CDs_curve = []
#######


distChamfer = dist_chamfer_3D.chamfer_3DDist()

for epoch in range(args.epoch):
    subnet1.train()
    encoder.train()
    # learning rate schedule
    if epoch == 200:
        optimizer.param_groups[0]['lr'] = args.lr/10
    if epoch == 300:
        optimizer.param_groups[0]['lr'] = args.lr/100
    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        img, points, normals, name, cat = data # img, vertices_ref, faces_ref, _ , _ 
        img, normals, points = img.cuda(), normals.cuda(), points.cuda()
        choice = np.random.choice(points.size(1), args.num_vertices, replace=False) 
        points_choice = points[:, choice, :].contiguous() 
        vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                        vertices_sphere.size(2)).contiguous()) # Shepre 
        ## Main flow ##
        # Encoder Img - Shape Fearures X 
        feature_img = encoder(img)
        # Subnet1 - Mesh Deform (vertices' offset) & Error Estimation
        pointsRec, pointsRec_samples, out_error_estimator, random_choice,_ = subnet1(args= args, img_featrue=feature_img, points=vertices_input, faces_cuda=faces_cuda, num_points= args.num_points, num_samples= args.num_samples, prune=False)
            
        ## losses ##
        # Chamfer distnace - ref vertices to predicted pertices
        dist1, dist2, _, idx2 = distChamfer(points_choice, pointsRec)
        CD_loss = torch.mean(dist1) + torch.mean(dist2)
        # Chamfer distnace - ref vertices to sampled faces of predicted vertices
        dist1_samples, dist2_samples, _, _ = distChamfer(points, pointsRec_samples.detach())
        CDs_loss = torch.mean(dist1_samples) + torch.mean(dist2_samples)
        # l2 loss 
        error_GT = torch.sqrt(dist2_samples.detach()[:,random_choice])
        l2_loss = mse_loss(out_error_estimator, error_GT.detach())
        # edge loss 
        edge_loss = get_edge_loss_stage1(pointsRec, edge_cuda.detach())
        # smoothnes_loss 
        smoothness_loss = get_smoothness_loss_stage1(pointsRec, parameters)
        # normal loss
        faces_cuda_bn = faces_cuda.unsqueeze(0).expand(pointsRec.size(0), faces_cuda.size(0),faces_cuda.size(1))
        normal_loss = get_normal_loss(pointsRec, faces_cuda_bn, normals, idx2)

        total_loss = CD_loss + l2_loss + 0.05 * edge_loss + (5e-7) * smoothness_loss \
                   + (1e-3) * normal_loss

        total_loss.backward()
        optimizer.step()  
        
        ### Averge and prints take from the origianl repo
        train_CD_loss.update(CD_loss.item())
        train_CDs_loss.update(CDs_loss.item())
        train_l2_loss.update(l2_loss.item())
        print('[%d: %d/%d] train_cd_loss:  %f , l2_loss: %f' % (epoch, i, len_dataset / args.batch_size,
                                                                     CD_loss.item(),l2_loss.item()))
    ### Averge and prints take from the origianl repo
    train_CD_curve.append(train_CD_loss.avg)
    train_CDs_curve.append(train_CDs_loss.avg)
    train_l2_curve.append(train_l2_loss.avg)
    train_CD_loss.reset()
    train_CDs_loss.reset()
    train_l2_loss.reset()

    # Validation step
    subnet1.eval()
    encoder.eval()
    for item in dataset_val.cat:
        dataset_val.perCatValueMeter[item].reset()
    with torch.no_grad():
        for i, data in enumerate(dataloader_val, 0):
            img, points, normals, name, cat = data
            img, normals, points = img.cuda(), normals.cuda(), points.cuda()
            choice = np.random.choice(points.size(1), args.num_vertices, replace=False) #TOOD - chagne take asis
            points_choice = points[:, choice, :].contiguous() #TOOD - chagne take asis
            vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                            vertices_sphere.size(2)).contiguous())
            ## Main flow ##
            # Encoder Img
            feature_img = encoder(img)
            #Subnet1
            pointsRec, pointsRec_samples, out_error_estimator, random_choice, _ = subnet1(args= args, img_featrue=feature_img, points=vertices_input, faces_cuda=faces_cuda, num_points= args.num_points, num_samples= args.num_samples, prune=False)
                
            ## losses ##
            # Chamfer distnace 
            dist1, dist2, _, idx2 = distChamfer(points_choice, pointsRec)
            CD_loss = torch.mean(dist1) + torch.mean(dist2)
            # Chamfer distnace 2  
            dist1_samples, dist2_samples, _, _ = distChamfer(points, pointsRec_samples.detach())
            CDs_loss = torch.mean(dist1_samples) + torch.mean(dist2_samples)
            # l2 loss 
            error_GT = torch.sqrt(dist2_samples.detach()[:,random_choice])
            l2_loss = mse_loss(out_error_estimator, error_GT.detach())
            # edge loss 
            edge_loss = get_edge_loss_stage1(pointsRec, edge_cuda.detach())
            # smoothnes_loss 
            smoothness_loss = get_smoothness_loss_stage1(pointsRec, parameters)
            # normal loss
            faces_cuda_bn = faces_cuda.unsqueeze(0).expand(pointsRec.size(0), faces_cuda.size(0),faces_cuda.size(1))
            normal_loss = get_normal_loss(pointsRec, faces_cuda_bn, normals, idx2)

            total_loss = CD_loss + l2_loss + 0.05 * edge_loss + (5e-7) * smoothness_loss \
                        + (1e-3) * normal_loss

            ### val mes
            val_CD_loss.update(CD_loss.item())
            dataset_val.perCatValueMeter[cat[0]].update(CDs_loss.item())
            val_l2_loss.update(l2_loss.item())
            val_CDs_loss.update(CDs_loss.item())
            print('[%d: %d/%d] val_cd_loss:  %f , l2_loss: %f' % (epoch, i, len(dataset_val) / args.batch_size,
                                                                  CD_loss.item(), l2_loss.item()))
                
            val_CD_curve.append(val_CD_loss.avg)
            val_l2_curve.append(val_l2_loss.avg)
            val_CDs_curve.append(val_CDs_loss.avg)


    log_table = {
        "train_CD_loss": train_CD_loss.avg,
        "val_CD_loss": val_CD_loss.avg,
        "train_l2_loss": train_l2_loss.avg,
        "val_l2_loss": val_l2_loss.avg,
        "val_CDs_loss": val_CDs_loss.avg,
        "epoch": epoch,
        "lr": args.lr,
    }


    print(log_table)
    for item in dataset_val.cat:
        print(item, dataset_val.perCatValueMeter[item].avg)
        log_table.update({item: dataset_val.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:  # open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
 
    torch.save(subnet1.state_dict(), '%s/subnet1.pth' % (dir_name))
    torch.save(encoder.state_dict(), '%s/encoder.pth' % (dir_name))
    val_CD_loss.reset()
    val_CDs_loss.reset()
    val_l2_loss.reset()
