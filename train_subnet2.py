## Trainer Partially based on the offical Repo of the paper - prints, general method, logger and so. 
import argparse
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as  dist_chamfer_3D
from Models.Main_models import Base_Img_to_Mesh as Base_network
from Models.Main_models import Subnet1 
from utils.utils import weights_init, AverageValueMeter, get_edges, create_round_spehere
from utils.loss import smoothness_loss_parameters, mse_loss, get_edge_loss, get_smoothness_loss, get_normal_loss # TODO - change names 
from utils.dataset import ShapeNet
import os, json
import torch
import torch.optim as optim
import numpy as np

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help=' batch size')
parser.add_argument('--workers', type=int, default=8,  help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=10000, help='number of points for GT')
parser.add_argument('--num_samples',type=int,default=2500, help='number of samples for error estimation')
parser.add_argument('--dir_name', type=str, default="subnet2", help='')
parser.add_argument('--lr',type=float,default=1e-3, help='initial learning rate')
parser.add_argument('--num_vertices', type=int, default=2562, help='number of vertices of the initial sphere')
parser.add_argument('--folder_path', type=str, default='./log/subnet1/', help='model path from the pretrained model')
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--device', type=int, default=0, help='GPU device')

args = parser.parse_args()
cuda = torch.device('cuda:{}'.format(args.device))

print(args)

## logger ##
dir_name = os.path.join('./log', args.dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')
with open(logname, 'a') as f:  # open and append
    f.write(str("subnet2") + '\n')

## data loader part take from the offical repo
dataset = ShapeNet(npoints=args.num_points, SVR=True, normal=True, train=True, class_choice='chair')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers))
dataset_val = ShapeNet(npoints=args.num_points, SVR=True, normal=True, train=False, class_choice='chair')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                              shuffle=False, num_workers=int(args.workers))
len_dataset = len(dataset)

# Create Round Spehere 
edge_cuda, vertices_sphere, faces_cuda, faces =  create_round_spehere(args.num_vertices, cuda = 'cuda:0')
parameters = smoothness_loss_parameters(faces)

## Load Models ##
# Img encoder 
encoder = Base_network().img_endoer
model_dict = encoder.state_dict()
pretrained_dict = {k: v for k, v in torch.load(args.folder_path+'/encoder.pth').items() if (k in model_dict)}
model_dict.update(pretrained_dict)
encoder.load_state_dict(model_dict)
encoder.to(cuda)

# Subnet1
subnet1 = Subnet1(cuda=cuda)
model_dict = subnet1.state_dict()
pretrained_dict = {k: v for k, v in torch.load(args.folder_path+'/subnet1.pth').items() if (k in model_dict)}
model_dict.update(pretrained_dict)
subnet1.load_state_dict(model_dict)
subnet1.to(cuda)

# Subnet2
subnet2 = Subnet1(cuda=cuda)
model_dict = subnet2.state_dict()
pretrained_dict = {k: v for k, v in torch.load(args.folder_path+'/subnet1.pth').items() if (k in model_dict)}
model_dict.update(pretrained_dict)
subnet2.load_state_dict(model_dict)
subnet2.to(cuda)

#optimizer load
optimizer = optim.Adam([
    {'params': subnet2.parameters()}
], lr=args.lr)

### Averge and prints take from the origianl repo
train_l2_loss = AverageValueMeter()
val_l2_loss = AverageValueMeter()
train_CDs_stage2_loss = AverageValueMeter()
val_CDs_stage2_loss = AverageValueMeter()
train_l2_curve = []
val_l2_curve = []
train_CDs_stage2_curve = []
val_CDs_stage2_curve = []
#######

distChamfer = dist_chamfer_3D.chamfer_3DDist()

for epoch in range(args.epoch):
    subnet1.eval()
    encoder.eval()
    subnet2.train()
    
    # learning rate schedule
    if epoch == 100:
        optimizer.param_groups[0]['lr'] = args.lr/10

    for i, data in enumerate(dataloader, 0):
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        img, points, normals, name, cat = data
        img, normals, points = img.to(cuda), normals.to(cuda), points.to(cuda) 
        choice = np.random.choice(points.size(1), args.num_vertices, replace=False) # Take ASIS origin repo
        points_choice = points[:, choice, :].contiguous() # Take ASIS origin repo
        vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                        vertices_sphere.size(2)).contiguous())
        ## main flow ##

        with torch.no_grad():
            # Encoder Img
            feature_img = encoder(img)
            #Subnet1
            pointsRec, pointsRec_samples, out_error_estimator, random_choice, faces_cuda_bn = subnet1(args= args, img_featrue=feature_img, points=vertices_input, faces_cuda=faces_cuda, num_points= args.num_points, num_samples= args.num_samples, prune= True)

        #Subnet2 - trained 
        pointsRec2, pointsRec_samples2, out_error_estimator2, random_choice2, _ = subnet2(args= args, img_featrue=feature_img, points=pointsRec, faces_cuda=faces_cuda_bn, num_points= args.num_points, num_samples= args.num_samples, prune= False)

        ## losses ##
        _, _, _, idx2 = distChamfer(points, pointsRec2)
        # Chamfer distnace 2  
        dist1_samples, dist2_samples, _, _ = distChamfer(points, pointsRec_samples2)
        CDs_loss_stage2 = torch.mean(dist1_samples) + torch.mean(dist2_samples)
        # l2 loss 
        error_GT = torch.sqrt(dist2_samples.detach()[:,random_choice2])
        l2_loss = mse_loss(out_error_estimator2, error_GT.detach())
        # edge loss 
        edge_loss = get_edge_loss(pointsRec2, faces_cuda_bn)
        # smoothnes_loss 
        smoothness_loss = get_smoothness_loss(pointsRec2, parameters, faces_cuda_bn)
        # normal loss
        normal_loss = get_normal_loss(pointsRec2, faces_cuda_bn, normals, idx2)
        total_loss = CDs_loss_stage2 + l2_loss + 0.05 * edge_loss + (2e-7) * smoothness_loss \
                   + (5e-3) * normal_loss

        total_loss = CDs_loss_stage2

        total_loss.backward()
        optimizer.step()  
        ###### 
        train_CDs_stage2_loss.update(CDs_loss_stage2.item())
        train_l2_loss.update(l2_loss.item())
        print('[%d: %d/%d] train_cd_loss:  %f  / l2_loss: %f' % (epoch, i, len_dataset / args.batch_size,
                                                                 CDs_loss_stage2.item(),l2_loss.item()))
        
    train_l2_curve.append(train_l2_loss.avg)
    train_CDs_stage2_curve.append(train_CDs_stage2_loss.avg)
    train_CDs_stage2_loss.reset()
    train_l2_loss.reset()

    ## eval step
    subnet1.eval()
    subnet2.eval()
    encoder.eval()

    for item in dataset_val.cat:
        dataset_val.perCatValueMeter[item].reset()
    with torch.no_grad():
        for i, data in enumerate(dataloader_val, 0):
            optimizer.zero_grad()
            img, points, normals, name, cat = data
            img, normals, points = img.to(cuda), normals.to(cuda), points.to(cuda)
            choice = np.random.choice(points.size(1), args.num_vertices, replace=False) # Take ASIS origin repo
            points_choice = points[:, choice, :].contiguous() # Take ASIS origin repo
            vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                            vertices_sphere.size(2)).contiguous())
            ## Main flow ## 
            # Encoder Img
            feature_img = encoder(img)
            #Subnet1
            pointsRec, pointsRec_samples, out_error_estimator, random_choice, faces_cuda_bn = subnet1(args= args, img_featrue=feature_img, points=vertices_input, faces_cuda=faces_cuda, num_points= args.num_points, num_samples= args.num_samples, prune= True)
            #Subnet2
            pointsRec2, pointsRec_samples2, out_error_estimator2, random_choice2, _ = subnet2(args= args, img_featrue=feature_img, points=pointsRec, faces_cuda=faces_cuda_bn, num_points= args.num_points, num_samples= args.num_samples, prune= False)
                
            ## losses ##
            _, _, _, idx2 = distChamfer(points, pointsRec2)
            # Chamfer distnace 2  
            dist1_samples, dist2_samples, _, _ = distChamfer(points, pointsRec_samples2)
            CDs_loss_stage2 = torch.mean(dist1_samples) + torch.mean(dist2_samples)
            # l2 loss 
            error_GT = torch.sqrt(dist2_samples.detach()[:,random_choice2])
            l2_loss = mse_loss(out_error_estimator2, error_GT.detach())
            # edge loss 
            edge_loss = get_edge_loss(pointsRec2, faces_cuda_bn)
            # smoothnes_loss 
            smoothness_loss = get_smoothness_loss(pointsRec2, parameters, faces_cuda_bn)
            # normal loss
            #faces_cuda_bn = faces_cuda.unsqueeze(0).expand(pointsRec.size(0), faces_cuda.size(0),faces_cuda.size(1))
            normal_loss = get_normal_loss(pointsRec2, faces_cuda_bn, normals, idx2)

            total_loss = CDs_loss_stage2 + l2_loss + 0.05 * edge_loss + (5e-7) * smoothness_loss \
                    + (1e-3) * normal_loss
            
            ### val mes
            val_CDs_stage2_loss.update(CDs_loss_stage2.item())
            val_l2_loss.update(l2_loss.item())
            dataset_val.perCatValueMeter[cat[0]].update(CDs_loss_stage2.item())
            
            print('[%d: %d/%d] val_cd_loss:  %f , l2_loss: %f' % (epoch, i, len(dataset_val) / args.batch_size,
                                                                    CDs_loss_stage2.item(), l2_loss.item()))
                
            val_l2_curve.append(val_l2_loss.avg)
            val_CDs_stage2_curve.append(val_CDs_stage2_loss.avg)


    log_table = {
        "train_l2_loss": train_l2_loss.avg,
        "train_cds_stage2": train_CDs_stage2_loss.avg,
        "val_l2_loss": val_l2_loss.avg,
        "val_cds_stage2": val_CDs_stage2_loss.avg,
        "epoch": epoch,
        "lr": args.lr,
    }

    print(log_table)
    for item in dataset_val.cat:
        print(item, dataset_val.perCatValueMeter[item].avg)
        log_table.update({item: dataset_val.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:  # open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
 
    torch.save(subnet2.state_dict(), '%s/subnet2.pth' % (dir_name))
    val_CDs_stage2_loss.reset()
    val_l2_loss.reset()