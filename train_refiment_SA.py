from __future__ import print_function
import argparse
import sys
sys.path.insert(1, '/data/ashomer/project/TMNet') # TODO - fix Chamfer distance compile in this folder
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as  dist_chamfer_3D
sys.path.insert(1, '/data/ashomer/project/Project_DMR') # TODO - fix Chamfer distance compile in this folder
from Models.Main_models import Base_Img_to_Mesh as Base_network
from Models.Main_models import Subnet1 , DeformNet, Refinement

from utils.utils import weights_init, AverageValueMeter, get_edges, create_round_spehere, final_refined_mesh, samples_random
from utils.loss import smoothness_loss_parameters, mse_loss, get_edge_loss, get_smoothness_loss, get_normal_loss # TODO - change names 

from utils.dataset import ShapeNet
import random, os, json, sys
import torch
import torch.optim as optim
import scipy 
import numpy as np
torch.cuda.empty_cache()
random.seed(6185)
torch.manual_seed(6185)



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help=' batch size')
parser.add_argument('--workers', type=int, default=8,  help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=10000, help='number of points for GT')
parser.add_argument('--num_samples',type=int,default=2500, help='number of samples for error estimation')

parser.add_argument('--super_points', type=int, default=2500,
                    help='number of input points to pointNet, not used by default')
parser.add_argument('--dir_name', type=str, default="refinement", help='')
parser.add_argument('--lr',type=float,default=1e-3, help='initial learning rate')
parser.add_argument('--num_vertices', type=int, default=2562, help='number of vertices of the initial sphere')
parser.add_argument('--folder_path_subnet1', type=str, default='./log/subnet1/', help='model path from the pretrained model')
parser.add_argument('--folder_path_subnet2', type=str, default='./log/subnet2/', help='model path from the pretrained model')

parser.add_argument('--tau', type=float, default=0.1)

parser.add_argument('--device', type=int, default=1, help='GPU device')

# parser.add_argument('--manualSeed', type=int, default=6185)
args = parser.parse_args()
cuda = torch.device('cuda:{}'.format(args.device))

print(args)


dir_name = os.path.join('./log', args.dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

logname = os.path.join(dir_name, 'log.txt')
# blue = lambda x: '\033[94m' + x + '\033[0m'
# print("Random Seed: ", args.manualSeed)


dataset = ShapeNet(npoints=args.num_points, SVR=True, normal=True, train=True, class_choice='chair')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers))
dataset_val = ShapeNet(npoints=args.num_points, SVR=True, normal=True, train=False, class_choice='chair')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                              shuffle=False, num_workers=int(args.workers))

print('training set', len(dataset.datapath))
print('testing set', len(dataset_val.datapath))
len_dataset = len(dataset)

# Create Round Spehere 
edge_cuda, vertices_sphere, faces_cuda, faces =  create_round_spehere(args.num_vertices, cuda = cuda):

## Load Models ##
# Img encoder 
encoder = Base_network().img_endoer
model_dict = encoder.state_dict()
pretrained_dict = {k: v for k, v in torch.load(args.folder_path_subnet1+'/encoder.pth').items() if (k in model_dict)}
model_dict.update(pretrained_dict)
encoder.load_state_dict(model_dict)
encoder.to(cuda)

# Subnet1
subnet1 = Subnet1(cuda=cuda)
model_dict = subnet1.state_dict()
pretrained_dict = {k: v for k, v in torch.load(args.folder_path_subnet1+'/subnet1.pth').items() if (k in model_dict)}
model_dict.update(pretrained_dict)
subnet1.load_state_dict(model_dict)
subnet1.to(cuda)

# Subnet2
subnet2 = Subnet1(cuda=cuda)
model_dict = subnet2.state_dict()
pretrained_dict = {k: v for k, v in torch.load(args.folder_path_subnet2+'/subnet2.pth').items() if (k in model_dict)}
model_dict.update(pretrained_dict)
subnet2.load_state_dict(model_dict)
subnet2.to(cuda)

# Refine
refinement = Refinement(cuda=cuda) # Deform and Refine build the same
refinement.to(cuda)

optimizer = optim.Adam([
    {'params': refinement.parameters()}
], lr=args.lr)

# meters to record stats on learning
train_CDs_stage3_loss = AverageValueMeter()
val_CDs_stage3_loss = AverageValueMeter()
train_boundary_loss = AverageValueMeter()
val_boundary_loss = AverageValueMeter()
train_displace_loss = AverageValueMeter()
val_displace_loss = AverageValueMeter()


with open(logname, 'a') as f:  # open and append
    f.write(str(refinement) + '\n')

train_CDs_stage3_curve = []
val_CDs_stage3_curve = []
train_boundary_curve = []
val_boundary_curve = []
train_displace_curve = []
val_displace_curve = []

distChamfer = dist_chamfer_3D.chamfer_3DDist()

for epoch in range(args.epoch):
    # TRAIN MODE
    train_CDs_stage3_loss.reset()
    train_boundary_loss.reset()
    train_displace_loss.reset()

    subnet1.eval()
    encoder.eval()
    subnet2.eval()
    refinement.train()
    # learning rate schedule
    
    if epoch == 100:
        optimizer.param_groups[0]['lr'] = args.lr/10


    for i, data in enumerate(dataloader, 0):
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        img, points, normals, name, cat = data
        img, normals, points = img.to(cuda), normals.to(cuda), points.to(cuda)
        choice = np.random.choice(points.size(1), args.num_vertices, replace=False) #TOOD - chagne take asis
        points_choice = points[:, choice, :].contiguous() #TOOD - chagne take asis
        vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                        vertices_sphere.size(2)).contiguous())
        
        with torch.no_grad():
                    
            # Encoder Img
            feature_img = encoder(img)
            
            #Subnet1
            pointsRec, pointsRec_samples, out_error_estimator, random_choice, faces_cuda_bn = subnet1(args= args, img_featrue=feature_img, points=vertices_input, faces_cuda=faces_cuda, num_points= args.num_points, num_samples= args.num_samples, prune= True)

            #Subnet2
            pointsRec2, pointsRec_samples2, out_error_estimator2, random_choice2, faces_cuda_bn2 = subnet2(args= args, img_featrue=feature_img, points=pointsRec, faces_cuda=faces_cuda_bn, num_points= args.num_points, num_samples= args.num_samples, prune= True, tau = args.tau/2)
                
            #Refiment 
            pointsRec3_boundary, displace_loss, selected_pair_all, selected_pair_all_len = refinement(points = pointsRec2, img_featrue = feature_img, faces_cuda_bn = faces_cuda_bn2 )
            pointsRec3 = final_refined_mesh(selected_pair_all, selected_pair_all_len, pointsRec3_boundary, pointsRec2, batch_size = img.shape[0] )
            pointsRec3_samples, _ = samples_random(faces_cuda_bn, pointsRec3, args.num_points, device = cuda)

        
        ## losses ##
        # Chamfer distnace 3
        dist13_samples, dist23_samples, _, _ = distChamfer(points, pointsRec3_samples)
        cds_stage3 = (torch.mean(dist13_samples)) + (torch.mean(dist23_samples))
        
        # boundery
        points_select = pointsRec3.index_select(1, selected_pair_all.view(-1)).\
            view(pointsRec3.size(0) * selected_pair_all.size(0),selected_pair_all.size(1), selected_pair_all.size(2),
                 pointsRec3.size(2))
        indices = (torch.arange(0, faces_cuda_bn.size(0)) * (1 + faces_cuda_bn.size(0))).type(torch.cuda.LongTensor)
        points_select = points_select.index_select(0, indices)
        edge_a = points_select[:, :, 0] - points_select[:, :, 1]
        edge_b = points_select[:, :, 0] - points_select[:, :, 2]
        edge_a_norm = edge_a / (torch.norm((edge_a + 1e-6), dim=2)).unsqueeze(2)
        edge_b_norm = edge_b / (torch.norm((edge_b + 1e-6), dim=2)).unsqueeze(2)
        final = torch.abs(edge_a_norm + edge_b_norm).sum(2)
        loss_mask = (selected_pair_all.sum(2) != 0).type(torch.cuda.FloatTensor).detach()
        loss_boundary_final = (final * loss_mask).sum() / len(loss_mask.nonzero())
 


        # Total loss
        total_loss = cds_stage3 + 0.5 * loss_boundary_final + 0.2 * displace_loss


        total_loss.backward()
        optimizer.step()  
        ###### 
        train_CDs_stage3_loss.update(cds_stage3.item())
        train_boundary_loss.update(loss_boundary_final.item())
        train_displace_loss.update(displace_loss.item())
        print('[%d: %d/%d] train_cd_loss:  %f' % (epoch, i, len_dataset / args.batch_size, cds_stage3.item()))
        
    # UPDATE CURVES

    train_CDs_stage3_curve.append(train_CDs_stage3_loss.avg)
    train_boundary_curve.append(train_boundary_loss.avg)
    train_displace_curve.append(train_displace_loss.avg)

    # VALIDATION
    subnet1.eval()
    subnet2.eval()
    encoder.eval()
    refinement.eval()

    val_CDs_stage3_loss.reset()
    val_boundary_loss.reset()
    val_displace_loss.reset()

    for item in dataset_val.cat:
        dataset_val.perCatValueMeter[item].reset()
        for i, data in enumerate(dataloader_val, 0):
            torch.cuda.empty_cache()

            img, points, normals, name, cat = data
            img, normals, points = img.to(cuda), normals.to(cuda), points.to(cuda)
            choice = np.random.choice(points.size(1), args.num_vertices, replace=False) #TOOD - chagne take asis
            points_choice = points[:, choice, :].contiguous() #TOOD - chagne take asis
            vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                            vertices_sphere.size(2)).contiguous())
            
            with torch.no_grad():  
                # Encoder Img
                feature_img = encoder(img)
                #Subnet1
                pointsRec, pointsRec_samples, out_error_estimator, random_choice, faces_cuda_bn = subnet1(args= args, img_featrue=feature_img, points=vertices_input, faces_cuda=faces_cuda, num_points= args.num_points, num_samples= args.num_samples, prune= True)
                #Subnet2
                pointsRec2, pointsRec_samples2, out_error_estimator2, random_choice2, faces_cuda_bn2 = subnet2(args= args, img_featrue=feature_img, points=pointsRec, faces_cuda=faces_cuda_bn, num_points= args.num_points, num_samples= args.num_samples, prune= True, tau = args.tau/2)
                #Refiment 
                pointsRec3_boundary, displace_loss, selected_pair_all, selected_pair_all_len = refinement(points = pointsRec2, img_featrue = feature_img, faces_cuda_bn = faces_cuda_bn2 )
                pointsRec3 = final_refined_mesh(selected_pair_all, selected_pair_all_len, pointsRec3_boundary, pointsRec2, batch_size = img.shape[0])
                pointsRec3_samples, _ = samples_random(faces_cuda_bn, pointsRec3, args.num_points, device = cuda)

            
            ## losses ##
            # Chamfer distnace 3
            dist13_samples, dist23_samples, _, _ = distChamfer(points, pointsRec3_samples)
            cds_stage3 = (torch.mean(dist13_samples)) + (torch.mean(dist23_samples))
            
            # boundery
            points_select = pointsRec3.index_select(1, selected_pair_all.view(-1)).\
                view(pointsRec3.size(0) * selected_pair_all.size(0),selected_pair_all.size(1), selected_pair_all.size(2),
                    pointsRec3.size(2))
            indices = (torch.arange(0, faces_cuda_bn.size(0)) * (1 + faces_cuda_bn.size(0))).type(torch.cuda.LongTensor)
            points_select = points_select.index_select(0, indices)
            edge_a = points_select[:, :, 0] - points_select[:, :, 1]
            edge_b = points_select[:, :, 0] - points_select[:, :, 2]
            edge_a_norm = edge_a / (torch.norm((edge_a + 1e-6), dim=2)).unsqueeze(2)
            edge_b_norm = edge_b / (torch.norm((edge_b + 1e-6), dim=2)).unsqueeze(2)
            final = torch.abs(edge_a_norm + edge_b_norm).sum(2)
            loss_mask = (selected_pair_all.sum(2) != 0).type(torch.cuda.FloatTensor).detach()
            loss_boundary_final = (final * loss_mask).sum() / len(loss_mask.nonzero())

            # Total loss
            total_loss = cds_stage3 + 0.5 * loss_boundary_final + 0.2 * displace_loss

            val_CDs_stage3_loss.update(cds_stage3.item())
            val_boundary_loss.update(loss_boundary_final.item())
            val_displace_loss.update(displace_loss.item())
            dataset_val.perCatValueMeter[cat[0]].update(cds_stage3.item())
            print('[%d: %d/%d] val_cd_loss:  %f ' % (epoch, i, len(dataset_val)/args.batch_size, cds_stage3.item()))

        val_CDs_stage3_curve.append(val_CDs_stage3_loss.avg)
        val_boundary_curve.append(val_boundary_loss.avg)
        val_displace_curve.append(val_displace_loss.avg)
        

 

    log_table = {
    "train_cds_stage3_loss" : train_CDs_stage3_loss.avg,
    "val_cds_stage3_loss": val_CDs_stage3_loss.avg,
    "epoch": epoch,
    "lr": args.lr}

    print(log_table)
    for item in dataset_val.cat:
        print(item, dataset_val.perCatValueMeter[item].avg)
        log_table.update({item: dataset_val.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:  # open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
 
    torch.save(refinement.state_dict(), '%s/refinement.pth' % (dir_name))
