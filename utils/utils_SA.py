import os
import random
import numpy as np
import torch
import pandas as pd
from scipy.sparse import coo_matrix

def faces_selection(samples_num, device, faces):
    ''' Select faces to be sampled ''' 

    faces = faces.cpu().data.numpy() # [b, n_faces, 3 vertices, 3 coords]
    random_face_ind = np.sort(np.round(np.random.rand(faces.shape[0], samples_num) * faces.shape[1]), axis=1).astype(int) # rand indx in range (0,1) * num_face
    faces_index_tensor = (torch.from_numpy(random_face_ind).cuda()).type(torch.cuda.LongTensor).to(device)
    
    return faces_index_tensor


def faces_selection_orig(sampled_number, device, faces_points):
    ''' Select faces to be sampled by normal size ''' 

    faces = faces_points.cpu().data.numpy() # [b, n_faces, 3 vertices, 3 coords]
    v1, v2, v3 = faces[:, :, 0], faces[:, :, 1], faces[:, :, 2] # [b, n_faces, 3 coords]
    
    normal = np.cross(v2 - v1, v3 - v1) # cross-product triangles's vectors
    normal_size = np.sqrt(normal[:, :, 0] ** 2 + normal[:, :, 1] ** 2 + normal[:, :, 2] ** 2) # (x^2 + y^2 + z^2)^0.5 - [b, n_faces]
    normal_sum = np.sum(normal_size, axis=1) # normals sum
    normal_cum = np.cumsum(normal_size, axis=1) # normals cumsum

    # Select faces 
    faces_pick = normal_sum[:, np.newaxis] * np.random.random(sampled_number)[np.newaxis, :] # (b, samples_num)

    # Index of sampled points 
    faces_index = []
    for i in range(faces_pick.shape[0]):
        faces_index.append(np.searchsorted(normal_cum[i], faces_pick[i]))

    faces_index = np.array(faces_index)  
    faces_index = np.clip(faces_index,0,normal_cum.shape[1]-1)
    faces_index_tensor = (torch.from_numpy(faces_index).cuda()).type(torch.cuda.LongTensor).to(device)
    faces_index_tensor_sort = faces_index_tensor.sort(1)[0]
    return faces_index_tensor_sort


def samples_random(faces_cuda, pointsRec, sampled_number,device='cuda:0'): # TODO Sapir
    """ Random vertices sampleing on given faces """ 

    # TODO - understand 
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

    # Select triangles to be sampled
    faces_index_sample = faces_selection(sampled_number, device, faces_points)

    faces_sampled = faces_points[faces_index_sample].clone()
    v1 = faces_sampled[:, :, 0]
    v2_v3 = faces_sampled[:, :, 1:]
    tri_vectors = v2_v3 - v1.unsqueeze(2).expand_as(v2_v3)

    diag_index = (torch.arange(0, pointsRec.size()[0]).to(device))
    diag_index=diag_index.type(torch.cuda.LongTensor)
    diag_index = (1+faces_index_sample.size(0)) * diag_index

    v1 = v1.index_select(0, diag_index)
    tri_vectors = tri_vectors.index_select(0, diag_index)

    random_lenghts = ((torch.randn(pointsRec.size()[0], v1.size()[1], 2, 1).uniform_(0, 1)).to(device))
    random_test = random_lenghts.sum(2).squeeze(2) > 1.0
    random_test_minus = random_test.type(torch.cuda.FloatTensor).unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 1)
    random_lenghts = torch.abs(random_lenghts - random_test_minus)
    random_lenghts = random_lenghts.repeat(1,1,1,3)

    sample_vector = (tri_vectors * random_lenghts).sum(2)
    samples = sample_vector + v1

    return samples, faces_index_sample


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


def get_max(errors, sampled_face_index, fn=5120): # TODO Sapir
    """ Return the max error between all sampled points of a face """

    batch_size, samples_num = errors.shape[0], errors.shape[1]

    b = torch.stack([torch.bincount(x,minlength=fn).cumsum(0) for x in sampled_face_index])
    b2 = torch.zeros_like(b)
    b2[:,1:].copy_(b[:,:-1])
    c = torch.LongTensor(range(samples_num)).expand([batch_size,samples_num])

    sampled_face_index2 = c-torch.gather(b2, 1, sampled_face_index)

    max_errors = []
    for i in range(batch_size):
        row = sampled_face_index[i]
        col = sampled_face_index2[i]
        data = errors[i]
        coo = coo_matrix((data, (row, col)), shape=(fn, int(b[i].max())))
        max_errors.append(torch.from_numpy(coo.max(axis=1).toarray()))
    max_errors = torch.stack(max_errors)
    return max_errors


def prune(faces_cuda_bn, error, tau, sampled_faces_index, pool='max', faces_number=5120, device='cuda:0'):
    ''' Remove bounday faces with error larger than tau, unite triangles after pruning and remove open triangles'''

    faces_cuda_bn = faces_cuda_bn.clone()

    # Decrease tau by a constant factor
    tau = tau / 10.0 

    # Positive error
    error = torch.pow(error, 2)

    # For each face, get the max error of its samples
    face_error = get_max(error.cpu(), sampled_faces_index.cpu(), faces_number).squeeze(2).to(device)

    # Mark faces to prune a zero point
    faces_cuda_bn[face_error > tau] = 0

    faces_cuda_set = []
    for faces_cuda in faces_cuda_bn:
        boundary_edge = get_boundary(faces_cuda)
        boundary_edge_point = boundary_edge.astype(np.int64).reshape(-1)

        # Remove vertices that have more than 2 edges -  unite triangles 
        counts = pd.value_counts(boundary_edge_point)
        toremove_point = torch.from_numpy(np.array(counts[counts > 2].sampled_faces_index)).to(device)
        faces_cuda_expand = faces_cuda.unsqueeze(2).expand(faces_cuda.shape[0], faces_cuda.shape[1],
                                                           toremove_point.shape[0])
        toremove_point_expand = toremove_point.unsqueeze(0).unsqueeze(0).\
            expand(faces_cuda.shape[0],faces_cuda.shape[1],toremove_point.shape[0])
        toremove_index = ((toremove_point_expand == faces_cuda_expand).sum(2).sum(1)) != 0
        faces_cuda[toremove_index] = 0
        triangles = faces_cuda.cpu().data.numpy()

        # Remove vertices that have 2 edge - open triangles
        v = pd.value_counts(triangles.reshape(-1))
        v = v[v == 1].sampled_faces_index
        for vi in v:
            if np.argwhere(triangles == vi).shape[0] == 0:
                continue
            triangles[np.argwhere(triangles == vi)[0][0]] = 0

        faces_cuda_set.append(torch.from_numpy(triangles).to(device).unsqueeze(0))
    
    faces_cuda_bn = torch.cat(faces_cuda_set, 0)
    return faces_cuda_bn


def get_edges(faces):
    ''' Return edges list'''
    
    edge = []
    for i, j in enumerate(faces):
        edge1, edge2, edge3 = j[:2], j[1:], j[[0, 2]]
        edge.append(edge1).append(edge2).append(edge3)
    edge = np.array(edge)

    # Represent each edge with a scalar
    edge_im = edge[:, 0] * edge[:, 1] + (edge[:, 0] + edge[:, 1]) * 1j
    
    # Remove repetitions (unique scalars)
    edge_unique = edge[ np.unique(edge_im, return_index=True)[1]]
    
    edge_cuda = (torch.from_numpy(edge_unique).type(torch.cuda.LongTensor)).detach()
    return edge_cuda


def get_boundary(faces):
    ''' Boundary edges are not shared between faces, meaning there's only one face that include this edge '''

    # Face = v1, v2, v3
    vertices_num = faces.max().item() + 1
    faces_np = faces.cpu().data.numpy()
    faces_np = faces_np[faces_np.sum(1).nonzero()]

    # All Edges - each faces and its edges, shared edges listed twice 
    edge1 = faces_np[:, :2] # v2-->v1 
    edge2 = faces_np[:, [0, 2]] # v3-->v1
    edge3 = faces_np[:, 1:] # v3-->v2
    face_edges = np.concatenate((edge1, edge2, edge3), 0).sort(1)

    # Each edge is identified with a scale - (V1*2562+V2)
    face_edges_identifier = face_edges[:, 0] * vertices_num + face_edges[:, 1]

    # Find boundary edge list - edges that appear only once in the edges list
    edges_identifier_ind = np.arange(0, face_edges_identifier.shape[0])
    unique_edges_identifier_ind = np.unique(face_edges_identifier, return_index=True)[1]
    shared_edges_identifier_ind = np.array(list(set(edges_identifier_ind).difference(set(unique_edges_identifier_ind)))) # All edges without unique
    
    unique_edges_identifier = face_edges_identifier[unique_edges_identifier_ind]
    shared_edges_identifier = face_edges_identifier[shared_edges_identifier_ind]
    boundary_edges_identifier = np.array(list(set(unique_edges_identifier).difference(set(shared_edges_identifier)))) # Boundary = Unique without shared
    
    # Return to vertices depcitor for each edge  
    boundary_V1 = np.array(np.floor(boundary_edges_identifier / vertices_num))
    boundary_V2 = np.array(boundary_edges_identifier % vertices_num)
    boundary_edge = np.stack((boundary_V1, boundary_V2), 1) 

    # TODO Validate this part isn't relevant, return only boundary_edge
    #boundary_vertices = np.unique(np.concatenate((boundary_V1, boundary_V2), 0))
    #boundary_vertices = torch.from_numpy(boundary_vertices).type(torch.cuda.LongTensor)

    # For each boundary vertex, return its two edges 
    #boundary_edge_inverse = boundary_edge[:, [1, 0]]
    #boundary_edge_all = np.concatenate((boundary_edge, boundary_edge_inverse), 0)
    #boundary_edge_all = boundary_edge_all[np.argsort(boundary_edge_all[:, 0])]

    # Return boundary vertices  
    #selected_point = np.where(boundary_edge_all[:, 0] == np.concatenate((boundary_edge_all[1:, 0],
    #                                                                     boundary_edge_all[:1, 0]), 0))
    #boundary_pair = np.concatenate((boundary_edge_all[selected_point[0]],
    #                                boundary_edge_all[selected_point[0] + 1][:, 1:]), 1)
    #boundary_pair = torch.from_numpy(boundary_pair).type(torch.cuda.LongTensor)
    
    #return boundary_pair, boundary_vertices, boundary_edge

    return None, None, boundary_edge
