from __future__ import print_function
import argparse
#import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as  dist_chamfer_3D
import sys
sys.path.insert(1, '/data/ashomer/project/TMNet') # TODO - fix Chamfer distance compile in this folder
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as  dist_chamfer_3D
sys.path.insert(1, '/data/ashomer/project/Project_DMR') # TODO - fix Chamfer distance compile in this folder

from Models.Main_models import Base_Img_to_Mesh as Base_network
from utils.utils import weights_init, AverageValueMeter
from utils.dataset import ShapeNet
import random, os, json, sys
import torch
import torch.optim as optim

def self_schduler(epoch):
    if epoch == 100:
        optimizer = optim.Adam([
            {'params': network.point_cloud_encoder.parameters()},
            {'params': network.deformNet1.parameters()}
        ], lr=lrate/10.0)
    if epoch == 120:
        optimizer = optim.Adam(network.img_endoer.parameters(), lr=lrate)
    if epoch == 220:
        optimizer = optim.Adam(network.img_endoer.parameters(), lr=lrate / 10.0)
    return optimizer


random.seed(777)
torch.manual_seed(777)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help=' batch size')
parser.add_argument('--workers', type=int, default=8,  help='number of data loading workers')
parser.add_argument('--epoch', type=int, default=420, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=2500, help='number of points')
parser.add_argument('--super_points', type=int, default=2500,
                    help='number of input points to pointNet, not used by default')
parser.add_argument('--dir_name', type=str, default="base_weights_run", help='')
parser.add_argument('--lr',type=float,default=1e-3, help='initial learning rate')
# parser.add_argument('--manualSeed', type=int, default=6185)
args = parser.parse_args()
print(args)


dir_name = os.path.join('./log', args.dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

logname = os.path.join(dir_name, 'log.txt')

dataset = ShapeNet(npoints=args.num_points, SVR=True, normal=False, train=True, class_choice='chair')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.workers))
dataset_val = ShapeNet(npoints=args.num_points, SVR=True, normal=False, train=False, class_choice='chair')
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                                              shuffle=False, num_workers=int(args.workers))

print('training set', len(dataset.datapath))
print('testing set', len(dataset_val.datapath))
len_dataset = len(dataset)

# network = Pretrain(num_points=args.num_points)
network = Base_network()

network.cuda()  # put network on GPU

lrate = args.lr  # learning rate
optimizer = optim.Adam([
    {'params': network.point_cloud_encoder.parameters()},
    {'params': network.deformNet1.parameters()}
], lr=lrate)

# meters to record stats on learning
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()

with open(logname, 'a') as f:  # open and append
    f.write(str(network) + '\n')
# initialize learning curve on visdom, and color for each primitive in visdom display
train_list = []
val_list = []


distChamfer = dist_chamfer_3D.chamfer_3DDist()

for epoch in range(args.epoch):
    # TRAIN MODE
    train_loss.reset()
    network.train()
    # learning rate schedule
    optimizer = self_schduler(epoch)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        img, points, normals, name, cat = data
        img = img.cuda()
        points = points.transpose(2, 1).contiguous()
        points = points.cuda()
        # SUPER_RESOLUTION argsionally reduce the size of the points fed to PointNet
        points = points[:, :, :args.super_points].contiguous()
        # END SUPER RESOLUTION
        if epoch >= 120:
            pointsRec = network(img, mode='img')
        else:
            pointsRec = network(points)  # forward pass
        dist1, dist2,_,_ = distChamfer(points.transpose(2, 1).contiguous(), pointsRec)  # loss function
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_net.backward()
        train_loss.update(loss_net.item())
        optimizer.step()  # gradient update
      
        print('[%d: %d/%d] train loss:  %f ' % (epoch, i, len_dataset / args.batch_size, loss_net.item()))

    train_list.append(train_loss.avg)

    # VALIDATION
    val_loss.reset()
    for item in dataset_val.cat:
        dataset_val.perCatValueMeter[item].reset()

    network.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader_val, 0):
            img, points, normals, name, cat = data
            img = img.cuda()
            points = points.transpose(2, 1).contiguous()
            points = points.cuda()
            # SUPER_RESOLUTION
            points = points[:, :, :args.super_points].contiguous()
            # END SUPER RESOLUTION
            if epoch >= 120:
                pointsRec = network(img, mode='img')
            else:
                pointsRec = network(points)  # forward pass
            dist1, dist2,_,_ = distChamfer(points.transpose(2, 1).contiguous(), pointsRec)
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
            val_loss.update(loss_net.item())
            dataset_val.perCatValueMeter[cat[0]].update(loss_net.item())
            print('[%d: %d/%d] val loss:  %f ' % (epoch, i, len(dataset_val)/args.batch_size, loss_net.item()))
        # UPDATE CURVES
        val_list.append(val_loss.avg)


    log_table = {
        "train_loss": train_loss.avg,
        "val_loss": val_loss.avg,
        "epoch": epoch,
        "lr": lrate,
        "super_points": args.super_points,
    }
    print(log_table)
    for item in dataset_val.cat:
        print(item, dataset_val.perCatValueMeter[item].avg)
        log_table.update({item: dataset_val.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:  # open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')


    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
