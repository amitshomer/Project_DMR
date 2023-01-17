# Generally the plot mesh take from the offical repo and modified to support our models. 
import sys
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as  dist_chamfer_3D
from Models.Main_models import Base_Img_to_Mesh as Base_network
from Models.Main_models import Subnet1 , DeformNet, Refinement

from utils.utils import weights_init, AverageValueMeter, get_edges, prune, final_refined_mesh, samples_random, create_round_spehere
from utils.loss import smoothness_loss_parameters, mse_loss, get_edge_loss, get_smoothness_loss, get_normal_loss # TODO - change names 
from utils.mesh_plot_util import write_ply
from utils.dataset import ShapeNet
import random, os, sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

torch.cuda.empty_cache()
random.seed(6185)
torch.manual_seed(6185)



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help=' batch size')
parser.add_argument('--workers', type=int, default=8,  help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=10000, help='number of points for GT')
parser.add_argument('--num_samples',type=int,default=2500, help='number of samples for error estimation')

parser.add_argument('--dir_name', type=str, default="plot_mesh_folder", help='')
parser.add_argument('--lr',type=float,default=1e-3, help='initial learning rate')
parser.add_argument('--num_vertices', type=int, default=2562, help='number of vertices of the initial sphere')
parser.add_argument('--folder_path_subnet1', type=str, default='./log/subnet1/', help='model path from the pretrained model')
parser.add_argument('--folder_path_subnet2', type=str, default='./log/subnet2/', help='model path from the pretrained model')
parser.add_argument('--folder_refinement', type=str, default='./log/refinement/', help='model path from the pretrained model')


parser.add_argument('--tau', type=float, default=0.1)

parser.add_argument('--device', type=int, default=0, help='GPU device')

args = parser.parse_args()
cuda = torch.device('cuda:{}'.format(args.device))

print(args)


dir_name = os.path.join('./log', args.dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

logname = os.path.join(dir_name, 'log.txt')



dataset = ShapeNet(npoints=args.num_points, SVR=True, normal=True, train=True, class_choice='chair')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers))
dataset_val = ShapeNet(npoints=args.num_points, SVR=True, normal=True, train=False, class_choice='chair')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                              shuffle=False, num_workers=int(args.workers))

print('training set', len(dataset.datapath))
print('testing set', len(dataset_val.datapath))
len_dataset = len(dataset)

# Create Round Spehere - TODO take ASIS change
edge_cuda, vertices_sphere, faces_cuda, faces =  create_round_spehere(args.num_vertices, cuda = 'cuda:0')



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
model_dict = refinement.state_dict()
pretrained_dict = {k: v for k, v in torch.load(args.folder_refinement+'/refinement.pth').items() if (k in model_dict)}
model_dict.update(pretrained_dict)
refinement.load_state_dict(model_dict)
refinement.to(cuda)

with open(logname, 'a') as f:  # open and append
    f.write(str(refinement) + '\n')

distChamfer = dist_chamfer_3D.chamfer_3DDist()


subnet1.eval()
subnet2.eval()
encoder.eval()
refinement.eval()


for i, data in enumerate(dataloader_val, 0):
    torch.cuda.empty_cache()

    img, points, normals, name, cat = data
    cat =cat[0]
    name_fg = name[0]
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
        # pointsRec3_samples, _ = samples_random(faces_cuda_bn, pointsRec3, args.num_points, device = cuda)

        triangles_c1 = faces_cuda_bn[0].cpu().data.numpy()
        triangles_c2 = faces_cuda_bn2[0].cpu().data.numpy()
        triangles_c3 = faces_cuda_bn2[0].cpu().data.numpy()

        dir_category = dir_name + "/" + str(cat)
        if not os.path.exists(dir_category):
            os.mkdir(dir_category)

        b = np.zeros((np.shape(faces)[0],4)) + 3
        b[:,1:] = faces

        triangles_c1_tosave = triangles_c1[triangles_c1.sum(1).nonzero()[0]]
        b_c1 = np.zeros((np.shape(triangles_c1_tosave)[0],4)) + 3
        b_c1[:,1:] = triangles_c1_tosave
        triangles_c2_tosave = triangles_c2[triangles_c2.sum(1).nonzero()[0]]
        b_c2 = np.zeros((np.shape(triangles_c2_tosave)[0],4)) + 3
        b_c2[:,1:] = triangles_c2_tosave
        triangles_c3_tosave = triangles_c3[triangles_c3.sum(1).nonzero()[0]]
        b_c3 = np.zeros((np.shape(triangles_c3_tosave)[0],4)) + 3
        b_c3[:,1:] = triangles_c3_tosave

        write_ply(filename=dir_category + "/" + name_fg+"_GT",
                  points=pd.DataFrame(points.cpu().data.squeeze().numpy()), as_text=True)
        write_ply(filename=dir_category + "/" + name_fg+"_gen",
                  points=pd.DataFrame(pointsRec.cpu().data.squeeze().numpy()), as_text=True,
                  faces = pd.DataFrame(b.astype(int)))
        write_ply(filename=dir_category + "/" + name_fg+"_gen_pruned",
                  points=pd.DataFrame(pointsRec.cpu().data.squeeze().numpy()), as_text=True,
                  faces = pd.DataFrame(b_c1.astype(int)))
        write_ply(filename=dir_category + "/" + name_fg+"_gen2",
                    points=pd.DataFrame(pointsRec2.cpu().data.squeeze().numpy()), as_text=True,
                    faces = pd.DataFrame(b_c1.astype(int)))
        write_ply(filename=dir_category + "/" + name_fg+"_gen2_pruned",
                    points=pd.DataFrame(pointsRec2.cpu().data.squeeze().numpy()), as_text=True,
                    faces = pd.DataFrame(b_c2.astype(int)))
        write_ply(filename=dir_category+ "/" + name_fg+"_gen3",
                    points=pd.DataFrame(pointsRec3.cpu().data.squeeze().numpy()), as_text=True,
                    faces = pd.DataFrame(b_c3.astype(int)))