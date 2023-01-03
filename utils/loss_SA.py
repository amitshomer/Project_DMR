import torch
from torch import nn
import numpy as np


def normalize(input, p=2, dim=1, eps=1e-12): # TODO Sapir
    """ Measures the normal consistency between the generated mesh and ground truth """ 
    input = input / input.norm(p, dim)
    input = input.clamp(min=eps)
    input = input.unsqueeze(dim).expand_as(input)
    return input

 
def calculate_l2_loss(x, y):
    """ L2 loss """ 
    l2 = nn.MSELoss() 
    return l2(x,y)


def get_edge_loss(vertices, faces):
    """  
        Penalizes flying vertices, which ususally cause long edge - total edges legnth sum
        vertices - shape [b, 2562, 3]
        faces - shape [b, 5120, 3]
    """ 
    edge1, edge2, edge3 = faces[:,:,:2], faces[:,:,[0,2]] ,faces[:,:,1:]
    edges = torch.cat((edge1, edge2, edge3),1) # [b, 3*5120, 2]

    bs_edges, edges_num, edges_vertices_num,  = edges.size(0),edges.size(1),edges.size(2) # [b, 3*5120, 2-edge1,edge2]
    bs_vertices, vertex_coors_num = vertices.size(0), vertices.size(2)  # b, 3 (x,y,z)
    
    # For each edge, get vertices (x,y,z) coords - [b, 3*5120, 2, 3] 
    edges_vertices = vertices.index_select(1,edges.view(-1)).\
        view(bs_vertices*bs_edges, edges_num, edges_vertices_num, vertex_coors_num) # [b*b, 3*5120, 2, 3] 
    indices = (torch.arange(0,bs_vertices)*(1+bs_vertices)).type(torch.cuda.LongTensor)
    edges_vertices = edges_vertices.index_select(0,indices) # [b, 3*5120, 2, 3]

    # Sum edges length 
    edges_len = torch.norm((edges_vertices[:,:,0]-edges_vertices[:,:,1]), 2, 2)
    edges_len = torch.pow(edges_len, 2)
    nonzero = len(edges_len.nonzero()) # Returns a tensor containing the indices of all non-zero elements of input
    edge_loss = torch.sum(edges_len)/nonzero
    
    return edge_loss


def get_edge_loss_stage1(vertices, edge):
    """  
        Penalizes flying vertices, which ususally cause long edge - mean edge length
        vertices - shape [b, 2562, 3]
        edge - shape [b, 3*5120, 2]
    """ 
    edges_vertices = vertices.index_select(1,edge.view(-1)).\
        view(vertices.size(0),edge.size(0),edge.size(1),vertices.size(2))

    edges_len = torch.norm((edges_vertices[:,:,0]-edges_vertices[:,:,1]), 2, 2)
    edges_len = torch.pow(edges_len,2)
    edge_loss = torch.mean(edges_len)

    return edge_loss


def smoothness_loss_parameters(faces): # TODO Sapir
    """
        
        faces - shape [b, 5120, 3]
    """
    print('calculating the smoothness loss parameters, gonna take a few moments')
    
    if hasattr(faces, 'get'): faces = faces.get()
    
    edge1, edge2 = faces[:, 0:2], faces[:, 1:3]
    edges12 = list(set([tuple(v) for v in np.sort(np.concatenate((edge1, edge2), axis=0))]))
    v0s = np.array([v[0] for v in edges12], 'int32')
    v1s = np.array([v[1] for v in edges12], 'int32')

    v2s, v3s = [], []
    for v0, v1 in zip(v0s, v1s):
        count = 0
        for face in faces:
            if v0 in face and v1 in face:
                v = np.copy(face)
                v = v[v != v0]
                v = v[v != v1]
                if count == 0:
                    v2s.append(int(v[0]))
                    count += 1
                else:
                    v3s.append(int(v[0]))
        if len(v3s) < len(v2s):
            v3s.append(0)

    v2s = np.array(v2s, 'int32')
    v3s = np.array(v3s, 'int32')
    print('calculated')
    return v0s, v1s, v2s, v3s


def get_smoothness_loss_stage1(vertices, parameters, eps=1e-6): # TODO Sapir
    """ 
        Define a laplacian coordinate for each vertex
    """ 
    # make v0s, v1s, v2s, v3s
    # vertices (bs*num_points*3)
    v0s, v1s, v2s, v3s = parameters
    batch_size = vertices.size(0)

    v0s = torch.from_numpy(v0s).type(torch.cuda.LongTensor)
    v1s = torch.from_numpy(v1s).type(torch.cuda.LongTensor)
    v2s = torch.from_numpy(v2s).type(torch.cuda.LongTensor)
    v3s = torch.from_numpy(v3s).type(torch.cuda.LongTensor)

    v0s = vertices.index_select(1, v0s)
    v1s = vertices.index_select(1, v1s)
    v2s = vertices.index_select(1, v2s)
    v3s = vertices.index_select(1, v3s)

    a1 = v1s - v0s
    b1 = v2s - v0s
    a1l2 = torch.sum(a1.pow(2),dim=2)
    b1l2 = torch.sum(b1.pow(2),dim=2)
    a1l1 = torch.sqrt(a1l2 + eps)
    b1l1 = torch.sqrt(b1l2 + eps)
    ab1 = torch.sum(a1*b1,dim=2)

    cos1 = ab1 / (a1l1 * b1l1 + eps)
    sin1 = torch.sqrt(1 - cos1.pow(2) + eps)
    c1 = a1 * (((ab1/(a1l2+eps)).unsqueeze(2)).expand_as(a1))

    cb1 = b1 - c1
    cb1l1 = b1l1 * sin1

    a2 = v1s - v0s
    b2 = v3s - v0s
    a2l2 = torch.sum(a2.pow(2),dim=2)
    b2l2 = torch.sum(b2.pow(2),dim=2)
    a2l1 = torch.sqrt(a2l2 + eps)
    b2l1 = torch.sqrt(b2l2 + eps)
    ab2 = torch.sum(a2*b2,dim=2)

    cos2 = ab2 / (a2l1 * b2l1 + eps)
    sin2 = torch.sqrt(1 - cos2.pow(2) + eps)
    c2 = a2 * (((ab2 / (a2l2 + eps)).unsqueeze(2)).expand_as(a2))

    cb2 = b2 - c2
    cb2l1 = b2l1 * sin2

    cos = torch.sum(cb1*cb2, dim=2) / (cb1l1 * cb2l1 + eps)
    loss = torch.sum((cos+1).pow(2)) / batch_size

    return loss


def get_smoothness_loss(vertices, parameters,faces_bn,eps=1e-6): # TODO Sapir
    # make v0s, v1s, v2s, v3s
    # vertices (bs*num_points*3)
    v0s, v1s, v2s, v3s = parameters
    batch_size = vertices.size(0)

    v0s = torch.from_numpy(v0s).type(torch.cuda.LongTensor)
    v1s = torch.from_numpy(v1s).type(torch.cuda.LongTensor)
    v2s = torch.from_numpy(v2s).type(torch.cuda.LongTensor)
    v3s = torch.from_numpy(v3s).type(torch.cuda.LongTensor)

    vs = torch.stack((v0s,v1s,v2s,v3s),1)

    faces_bn_view = faces_bn.view(faces_bn.size(0),-1)
    faces_bn_view = faces_bn_view.sort(1)[0]

    count = torch.ones(1).expand_as(faces_bn_view).type(torch.cuda.LongTensor)
    count_sum = torch.zeros(1).expand((faces_bn.size(0),vertices.shape[1])).type(torch.cuda.LongTensor)

    count_sum = count_sum.scatter_add(1, faces_bn_view, count)
    count_sum = (count_sum > 0).type(torch.cuda.ByteTensor)

    b1 = count_sum.ne(1).type(torch.cuda.LongTensor)

    i2 = vs.expand([faces_bn.size(0),vs.size(0),vs.size(1)])
    i2_unrolled = i2.view(i2.size()[0],-1)
    out_mask = torch.gather(b1,1,i2_unrolled).resize_as_(i2)
    zero_mask = out_mask.sum(2,keepdim=True).long().eq(0).long().expand_as(out_mask)
    final = i2 * zero_mask

    v0s_bn = final[:,:,0]
    v1s_bn = final[:,:,1]
    v2s_bn = final[:,:,2]
    v3s_bn = final[:,:,3]

    v0s = vertices.index_select(1, v0s_bn.view(-1)).view(vertices.size(0)*v0s_bn.size(0),v0s_bn.size(1),vertices.size(2))
    v1s = vertices.index_select(1, v1s_bn.view(-1)).view(vertices.size(0)*v1s_bn.size(0),v1s_bn.size(1),vertices.size(2))
    v2s = vertices.index_select(1, v2s_bn.view(-1)).view(vertices.size(0)*v2s_bn.size(0),v2s_bn.size(1),vertices.size(2))
    v3s = vertices.index_select(1, v3s_bn.view(-1)).view(vertices.size(0)*v3s_bn.size(0),v3s_bn.size(1),vertices.size(2))

    indices = (torch.arange(0, vertices.size(0)) * (1 + vertices.size(0))).type(torch.cuda.LongTensor)

    v0s = v0s.index_select(0,indices)
    v1s = v1s.index_select(0,indices)
    v2s = v2s.index_select(0,indices)
    v3s = v3s.index_select(0,indices)

    a1 = v1s - v0s
    b1 = v2s - v0s
    a1l2 = torch.sum(a1.pow(2),dim=2)
    b1l2 = torch.sum(b1.pow(2),dim=2)
    a1l1 = torch.sqrt(a1l2 + eps)
    b1l1 = torch.sqrt(b1l2 + eps)
    ab1 = torch.sum(a1*b1,dim=2)

    cos1 = ab1 / (a1l1 * b1l1 + eps)
    sin1 = torch.sqrt(1 - cos1.pow(2) + eps)
    c1 = a1 * (((ab1/(a1l2+eps)).unsqueeze(2)).expand_as(a1))
    cb1 = b1 - c1
    cb1l1 = b1l1 * sin1

    a2 = v1s - v0s
    b2 = v3s - v0s
    a2l2 = torch.sum(a2.pow(2),dim=2)
    b2l2 = torch.sum(b2.pow(2),dim=2)
    a2l1 = torch.sqrt(a2l2 + eps)
    b2l1 = torch.sqrt(b2l2 + eps)
    ab2 = torch.sum(a2*b2,dim=2)
    cos2 = ab2 / (a2l1 * b2l1 + eps)
    sin2 = torch.sqrt(1 - cos2.pow(2) + eps)
    c2 = a2 * (((ab2 / (a2l2 + eps)).unsqueeze(2)).expand_as(a2))

    cb2 = b2 - c2
    cb2l1 = b2l1 * sin2

    cos = torch.sum(cb1*cb2, dim=2) / (cb1l1 * cb2l1 + eps)
    loss = torch.sum((cos+1).pow(2)) / batch_size
    return loss


def get_normal_loss(vertices, faces, gt_normals, idx2): # TODO Sapir
    """ 
            This loss requires the edge between a vertex with its neighbors
            to perpendicular to the observation from the ground truth
    """
    idx2 = idx2.type(torch.cuda.LongTensor).detach()
    edges = torch.cat((faces[:,:,:2],faces[:,:,[0,2]],faces[:,:,1:]),1)
    edges_vertices = vertices.index_select(1,edges.view(-1)).\
        view(vertices.size(0)*edges.size(0),edges.size(1),edges.size(2),vertices.size(2))
    indices = (torch.arange(0,vertices.size(0))*(1+vertices.size(0))).type(torch.cuda.LongTensor)
    edges_vertices = edges_vertices.index_select(0,indices)
    edges_len1 = edges_vertices[:,:,0] - edges_vertices[:,:,1]
    edges_len2 = edges_vertices[:,:,1] - edges_vertices[:,:,0]
    edges_vector = torch.stack((edges_len1,edges_len2),2)
    gt_normals = gt_normals.index_select(1, idx2.contiguous().view(-1)).contiguous().view(gt_normals.size(0) * idx2.size(0),
                                                                                          idx2.size(1), gt_normals.size(2))
    gt_normals = gt_normals.index_select(0, indices)
    gt_normals_edges = gt_normals.index_select(1, edges.view(-1)).view(gt_normals.size(0) * edges.size(0),
                                                                       edges.size(1), edges.size(2), gt_normals.size(2))
    gt_normals_edges = gt_normals_edges.index_select(0, indices)
    gt_normals_edges = normalize(gt_normals_edges, p=2, dim=3)
    edges_vector = normalize(edges_vector,p=2,dim=3)
    cosine = torch.abs(torch.sum(torch.mul(edges_vector, gt_normals_edges), 3))
    nonzero = len(cosine.nonzero())
    normal_loss = torch.sum(cosine)/nonzero

    return normal_loss

