import os
import random
import numpy as np
import torch
import pandas as pd
from scipy.sparse import coo_matrix
import utils.utils_SA as utils_sa

# initialize the weighs of the network for Convolutional layers and batchnorm layers
class AverageValueMeter(object): # TODO Sapir
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_max(errors, index, fn=5120): # TODO Sapir
    batch_size = errors.shape[0]
    number = errors.shape[1]
    b = torch.stack([torch.bincount(x,minlength=fn).cumsum(0) for x in index])
    b2 = torch.zeros_like(b)
    b2[:,1:].copy_(b[:,:-1])
    c = torch.LongTensor(range(number)).expand([batch_size,number])
    index2 = c-torch.gather(b2, 1, index)
    max_errors = []
    for i in range(batch_size):
        row = index[i]
        col = index2[i]
        data = errors[i]
        coo = coo_matrix((data, (row, col)), shape=(fn, int(b[i].max())))
        max_errors.append(torch.from_numpy(coo.max(axis=1).toarray()))
    max_errors = torch.stack(max_errors)
    return max_errors


def get_edges(faces):
    edge = []
    for i, j in enumerate(faces):
        edge.append(j[:2])
        edge.append(j[1:])
        edge.append(j[[0, 2]])
    edge = np.array(edge)
    edge_im = edge[:, 0] * edge[:, 1] + (edge[:, 0] + edge[:, 1]) * 1j
    unique = np.unique(edge_im, return_index=True)[1]
    edge_unique = edge[unique]
    edge_cuda = (torch.from_numpy(edge_unique).type(torch.cuda.LongTensor)).detach()
    return edge_cuda


def get_boundary(faces):
    vertices_number = faces.max().item() + 1 # Vertices amount
    
    triangles_new = faces.cpu().data.numpy()
    triangles_new = triangles_new[triangles_new.sum(1).nonzero()] # TODO not sure? 

    edges = np.concatenate((triangles_new[:, :2], triangles_new[:, [0, 2]], triangles_new[:, 1:]), 0) # array of edges
    edges.sort(1)
    edges_unrolled = edges[:, 0] * vertices_number + edges[:, 1]  # edges extended to 1d, represent edges as scaler
    # Find unique edges
    all_index = np.arange(0, edges_unrolled.shape[0]) # 14988 edges with replications 
    unique_index = np.unique(edges_unrolled, return_index=True)[1] # only 7543 unique edges 
    repeated_index = np.array(list(set(all_index).difference(set(unique_index)))) 
    unique_value = edges_unrolled[unique_index]
    repeated_value = edges_unrolled[repeated_index]
    # 
    boundary_value = np.array(list(set(unique_value).difference(set(repeated_value)))) # set - sequences of unique elements, difference - find the redunent for two sets 
    boundary_edge1 = np.array(np.floor(boundary_value / vertices_number))
    boundary_edge2 = np.array(boundary_value % vertices_number)
    boundary_edge = np.stack((boundary_edge1, boundary_edge2), 1)
    boundary_point = np.unique(np.concatenate((boundary_edge1, boundary_edge2), 0))
    boundary_edge_inverse = boundary_edge[:, [1, 0]]
    boundary_edge_all = np.concatenate((boundary_edge, boundary_edge_inverse), 0)
    boundary_edge_all = boundary_edge_all[np.argsort(boundary_edge_all[:, 0])]

    selected_point = np.where(boundary_edge_all[:, 0] == np.concatenate((boundary_edge_all[1:, 0],
                                                                         boundary_edge_all[:1, 0]), 0))
    selected_pair = np.concatenate((boundary_edge_all[selected_point[0]],
                                    boundary_edge_all[selected_point[0] + 1][:, 1:]), 1)
    selected_pair = torch.from_numpy(selected_pair).type(torch.cuda.LongTensor)
    boundary_point = torch.from_numpy(boundary_point).type(torch.cuda.LongTensor)

    return selected_pair, boundary_point, boundary_edge


def samples_random(faces_cuda, pointsRec, sampled_number,device='cuda:0'): # TODO Sapir

    if len(faces_cuda.size())==2:
        faces_points = pointsRec.index_select(1, faces_cuda.contiguous().view(-1)).contiguous().\
            view(pointsRec.size()[0], faces_cuda.size()[0], faces_cuda.size()[1], pointsRec.size()[2])
    elif len(faces_cuda.size())==3:
        faces_points = pointsRec.index_select(1, faces_cuda.contiguous().view(-1)).contiguous().\
            view(pointsRec.size()[0] ** 2, faces_cuda.size()[1], faces_cuda.size()[2], pointsRec.size()[2])
        index = (torch.arange(0, pointsRec.size()[0]) * (pointsRec.size()[0] + 1)).\
            type(torch.cuda.LongTensor).to(device)
        faces_points = faces_points.index_select(0,index)
    else:
        faces_points = None

    faces_points_np = faces_points.cpu().data.numpy() # (24, 5120, 3, 3)
    a = faces_points_np[:, :, 0] # (24, 5120, 3)
    b = faces_points_np[:, :, 1] # (24, 5120, 3)
    c = faces_points_np[:, :, 2] # (24, 5120, 3)

    cross = np.cross(b - a, c - a) # (24, 5120, 3)
    area = np.sqrt(cross[:, :, 0] ** 2 + cross[:, :, 1] ** 2 + cross[:, :, 2] ** 2) # (24, 5120)
    area_sum = np.sum(area, axis=1) # (24,)
    area_cum = np.cumsum(area, axis=1) # (24, 5120)
    faces_pick = area_sum[:, np.newaxis] * np.random.random(sampled_number)[np.newaxis, :]  # 32*7500. 10000 samples per image - (24,10000) TODO: why 32*7500?

    faces_index = []
    for i in range(faces_pick.shape[0]): # for each image in the batch, search the indice to add the samples insize the cumsum 
        faces_index.append(np.searchsorted(area_cum[i], faces_pick[i]))

    faces_index = np.array(faces_index)  # 32*7500 TODO: why 32*7500?. 24*10000
    faces_index = np.clip(faces_index,0,area_cum.shape[1]-1) 
    faces_index_tensor = (torch.from_numpy(faces_index).cuda()).type(torch.cuda.LongTensor).to(device)
    faces_index_tensor_sort = faces_index_tensor.sort(1)[0]

    tri_origins = faces_points[:, :, 0].clone()
    tri_vectors = faces_points[:, :, 1:].clone()
    tri_vectors = tri_vectors - tri_origins.unsqueeze(2).expand_as(tri_vectors)

    tri_origins = tri_origins.index_select(1, faces_index_tensor_sort.view(-1)).view(
        tri_origins.size()[0] * faces_index_tensor_sort.size()[0],
        faces_index_tensor_sort.size()[1], tri_origins.size()[2])
    tri_vectors = tri_vectors.index_select(1, faces_index_tensor_sort.view(-1)).view(
        tri_vectors.size()[0] * faces_index_tensor_sort.size()[0],
        faces_index_tensor_sort.size()[1], tri_vectors.size()[2], tri_vectors.size()[3])

    diag_index = (torch.arange(0, pointsRec.size()[0]).to(device))
    diag_index=diag_index.type(torch.cuda.LongTensor)
    diag_index = (1+faces_index_tensor_sort.size(0)) * diag_index

    tri_origins = tri_origins.index_select(0, diag_index)
    tri_vectors = tri_vectors.index_select(0, diag_index)

    random_lenghts = ((torch.randn(pointsRec.size()[0], tri_origins.size()[1], 2, 1).uniform_(0, 1)).to(device))
    random_test = random_lenghts.sum(2).squeeze(2) > 1.0
    random_test_minus = random_test.type(torch.cuda.FloatTensor).unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 1)
    random_lenghts = torch.abs(random_lenghts - random_test_minus)
    random_lenghts = random_lenghts.repeat(1,1,1,3)

    sample_vector = (tri_vectors * random_lenghts).sum(2)
    samples = sample_vector + tri_origins

    return samples, faces_index_tensor_sort


def get_boundary_points_bn(faces_cuda_bn, pointsRec_refined, device='cuda:0'):
    selected_pair_all = []
    selected_pair_all_len = []
    boundary_points_all = []
    boundary_points_all_len = []

    for bn in torch.arange(0, faces_cuda_bn.shape[0]):
        faces_each = faces_cuda_bn[bn]
        selected_pair, boundary_point, _ = get_boundary(faces_each)
        #selected_pair_sa, boundary_point_sa, aa_sa = utils_sa.get_boundary(faces_each.clone())
        #assert ((selected_pair != selected_pair_sa).sum()==0)
        #assert ((boundary_point != boundary_point_sa).sum()==0)
        #assert ((aa != aa_sa).sum()==0)
        selected_pair_all.append(selected_pair)
        selected_pair_all_len.append(len(selected_pair))
        boundary_points_all.append(boundary_point)
        boundary_points_all_len.append(len(boundary_point))

    max_len = np.array(selected_pair_all_len).max()
    max_len2 = np.array(boundary_points_all_len).max()
    for bn in torch.arange(0, faces_cuda_bn.shape[0]):
        if len(selected_pair_all[bn]) < max_len:
            len_cat = max_len - len(selected_pair_all[bn])
            tensor_cat = torch.zeros(len_cat, 3).type_as(selected_pair_all[bn])
            selected_pair_all[bn] = torch.cat((selected_pair_all[bn], tensor_cat), 0)
        if len(boundary_points_all[bn]) < max_len2:
            len_cat = max_len2 - len(boundary_points_all[bn])
            if len(boundary_points_all[bn]) > 0:
                tensor_cat = torch.Tensor(len_cat).fill_(boundary_points_all[bn][0]).type_as(boundary_points_all[bn]).to(device)
            else:
                tensor_cat = torch.zeros(len_cat).type_as(boundary_points_all[bn]).to(device)
            boundary_points_all[bn] = torch.cat((boundary_points_all[bn].to(device), tensor_cat), 0)

    selected_pair_all = torch.stack(selected_pair_all, 0)
    selected_pair_all_len = np.array(selected_pair_all_len)
    indices = (torch.arange(0, faces_cuda_bn.size(0)) * (1 + faces_cuda_bn.size(0))).type(torch.cuda.LongTensor).to(device)
    pointsRec_refined_boundary = pointsRec_refined.index_select(1, selected_pair_all.view(-1).to(device)). \
        view(pointsRec_refined.shape[0] * selected_pair_all.shape[0], selected_pair_all.shape[1],
             selected_pair_all.shape[2], pointsRec_refined.shape[2])
    pointsRec_refined_boundary = pointsRec_refined_boundary.index_select(0, indices)

    return pointsRec_refined_boundary, selected_pair_all, selected_pair_all_len


def prune(faces_cuda_bn, error, tau, index, pool='max', faces_number=5120, device='cuda:0'): # TODO Sapir
    error = torch.pow(error, 2) # erorr for each 10000 sampeled faces
    if not pool == 'sum': # decrease by factor
        tau = tau / 10.0
    ones = (torch.ones(1).to(device)).expand_as(error).type(torch.cuda.FloatTensor)
    zeros = (torch.Tensor(error.size(0) * faces_cuda_bn.size(1)).fill_(0)).to(device)
    index_1d = (index + (torch.arange(0, error.size(0)).unsqueeze(1).
                         expand_as(index).to(device) * faces_cuda_bn.size(1)).type(torch.cuda.LongTensor)).view(-1)
    face_error = zeros.index_add_(0, index_1d, error.view(-1)).view(error.size(0), faces_cuda_bn.size(1))
    face_count = zeros.index_add_(0, index_1d, ones.view(-1)).view(error.size(0), faces_cuda_bn.size(1))
    faces_cuda_bn = faces_cuda_bn.clone()

    if pool == 'mean': # normalize
        face_error = face_error / (face_count + 1e-12) 
    elif pool == 'max': # max value
        #face_error_sa = utils_sa.get_max(error.clone().cpu(), index.clone().cpu(), faces_number)
        face_error = get_max(error.cpu(), index.cpu(), faces_number)
        #assert ((face_error != face_error_sa).sum()==0)
        face_error = face_error.squeeze(2).to(device) 
    elif pool == 'sum': # the same
        face_error = face_error
    faces_cuda_bn[face_error > tau] = 0 # take only the faces with the small erorr

    faces_cuda_set = []
    for k in torch.arange(0, error.size(0)): # for each image in batch 
        faces_cuda = faces_cuda_bn[k]
        _, _, boundary_edge = get_boundary(faces_cuda) # get boundary edges 
        #, _, boundary_edge_sa = utils_sa.get_boundary(faces_cuda.clone())
        #assert ((boundary_edge != boundary_edge_sa).sum() == 0)
        boundary_edge_point = boundary_edge.astype(np.int64).reshape(-1) #get all vertices
        counts = pd.value_counts(boundary_edge_point) # count for each vertices how many times it apears
        toremove_point = torch.from_numpy(np.array(counts[counts > 2].index)).to(device) # remove vertices that apear more then two times (two edges from each vertex)
        faces_cuda_expand = faces_cuda.unsqueeze(2).expand(faces_cuda.shape[0], faces_cuda.shape[1],
                                                           toremove_point.shape[0])
        toremove_point_expand = toremove_point.unsqueeze(0).unsqueeze(0).\
            expand(faces_cuda.shape[0],faces_cuda.shape[1],toremove_point.shape[0])
        toremove_index = ((toremove_point_expand == faces_cuda_expand).sum(2).sum(1)) != 0 # remove faces
        faces_cuda[toremove_index] = 0
        triangles = faces_cuda.cpu().data.numpy()

        v = pd.value_counts(triangles.reshape(-1))
        v = v[v == 1].index # vertices that apears only once, needs to be removed - open triangles
        for vi in v:
            if np.argwhere(triangles == vi).shape[0] == 0:
                continue
            triangles[np.argwhere(triangles == vi)[0][0]] = 0

        faces_cuda_set.append(torch.from_numpy(triangles).to(device).unsqueeze(0))
    faces_cuda_bn = torch.cat(faces_cuda_set, 0)

    return faces_cuda_bn

def final_refined_mesh(selected_pair_all, selected_pair_all_len, pointsRec3_boundary, pointsRec2, batch_size):
    pointsRec3_set = []
    for ibatch in torch.arange(0, batch_size):
        length = selected_pair_all_len[ibatch]
        if length != 0:
            index_bp = selected_pair_all[ibatch][:, 0][:length]
            prb_final = pointsRec3_boundary[ibatch][:length]

            #print(prb_final)

            pr = pointsRec2[ibatch]
            index_bp = index_bp.view(index_bp.shape[0], -1).expand([index_bp.shape[0], 3])
            #pr_final = pr.scatter(dim=0, index=index_bp, source=prb_final)
            pr_final = pr.scatter(0, index_bp, prb_final)

            pointsRec3_set.append(pr_final)
        else:
            pr = pointsRec2[ibatch]
            pr_final = pr
            pointsRec3_set.append(pr_final)
    pointsRec3 = torch.stack(pointsRec3_set, 0)
    return pointsRec3