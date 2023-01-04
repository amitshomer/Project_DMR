import torch
from torch import nn
import numpy as np


# Smoomthness 
def smoothness_loss_parameters_sa(faces):
    """
        Returns a list of 4 neighbours  
        faces - shape [b, 5120, 3]
    """
    print('calculating the smoothness loss parameters, gonna take a few moments')
    
    if hasattr(faces, 'get'): faces = faces.get()
    
    # Extract edge vertices from faces
    edge1, edge2 = faces[:, 0:2], faces[:, 1:3]
    edges12 = list(set([tuple(v) for v in np.sort(np.concatenate((edge1, edge2), axis=0))]))
    
    # v0s and v1s are the first and the second vertices that define an edge - there are replications here
    v0s = np.array([v[0] for v in edges12], 'int32')
    v1s = np.array([v[1] for v in edges12], 'int32')
    
    # Initialize v2s and v3s
    v2s, v3s = [], []
    
    # Iterate over each edge 
    for v0, v1 in zip(v0s, v1s):
        count = 0 # count if this edge included in more than one face
        # Find faces that contain this edge
        for face in faces:
            if v0 in face and v1 in face:
                face_vertices = np.copy(face)
                # Remove this edge (v0,v1) from this face
                face_vertices = face_vertices[face_vertices != v0]
                face_vertices = face_vertices[face_vertices != v1]
                # First face containig this edge - first neighbour vertex
                if count == 0:
                    v2s.append(int(face_vertices[0]))
                    count += 1
                else: # Second face containing this edge - second neighbours vertex
                    v3s.append(int(face_vertices[0]))
        # Append 0 to v3s if this edge included in one face only, no neighbour
        if len(v3s) < len(v2s):
            v3s.append(0)
    
    v2s = np.array(v2s, 'int32')
    v3s = np.array(v3s, 'int32')
    print('smoothness loss parameters are calculated')
    return v0s, v1s, v2s, v3s


def _compute_loss(a, b, eps=1e-6):
    a_l2 = torch.sum(a.pow(2), dim=2)
    b_l2 = torch.sum(b.pow(2), dim=2)

    a_l1 = torch.sqrt(a_l2 + eps)
    b_l1 = torch.sqrt(b_l2 + eps)
    
    ab = torch.sum(a * b, dim=2)
    cos = ab / (a_l1 * b_l1 + eps)
    sin = torch.sqrt(1 - cos.pow(2) + eps)

    c = a * (((ab / (a_l2 + eps)).unsqueeze(2)).expand_as(a))
    cb = b - c
    cb_l1 = b_l1 * sin
    return cb, cb_l1


def get_smoothness_loss(vertices, parameters, faces_bn, eps=1e-6):
    # Convert params to tensors
    v0s, v1s, v2s, v3s = [torch.from_numpy(param).type(torch.cuda.LongTensor) for param in parameters]
    vs = torch.stack((v0s, v1s, v2s, v3s), 1)
    batch_size = vertices.size(0)

    # Reshape faces_bn and sort it
    faces_bn_view = faces_bn.view(faces_bn.size(0), -1)
    faces_bn_view = faces_bn_view.sort(1)[0]

    # Compute a count of the number of times each vertex appears in faces_bn
    count = torch.ones(1).expand_as(faces_bn_view).type(torch.cuda.LongTensor)
    count_sum = torch.zeros(1).expand((faces_bn.size(0), vertices.shape[1])).type(torch.cuda.LongTensor)
    count_sum = count_sum.scatter_add(1, faces_bn_view, count)
    count_sum = (count_sum > 0).type(torch.cuda.ByteTensor)

    # Create a mask for elements of vs that should be kept
    b1 = count_sum.ne(1).type(torch.cuda.LongTensor)
    i2 = vs.expand([faces_bn.size(0), vs.size(0), vs.size(1)])
    i2_unrolled = i2.view(i2.size()[0], -1)
    out_mask = torch.gather(b1, 1, i2_unrolled).resize_as_(i2)
    zero_mask = out_mask.sum(2, keepdim=True).long().eq(0).long().expand_as(out_mask)
    final = i2 * zero_mask

    # Select rows from the vertices tensor using the indices stored in final
    v0s_bn = final[:,:,0]
    v1s_bn = final[:,:,1]
    v2s_bn = final[:,:,2]
    v3s_bn = final[:,:,3]

    # Reshape the vertices tensor and select the rows using the indices
    v0s = vertices.index_select(1, v0s_bn.view(-1)).view(vertices.size(0) * v0s_bn.size(0), v0s_bn.size(1), vertices.size(2))
    v1s = vertices.index_select(1, v1s_bn.view(-1)).view(vertices.size(0) * v1s_bn.size(0), v1s_bn.size(1), vertices.size(2))
    v2s = vertices.index_select(1, v2s_bn.view(-1)).view(vertices.size(0) * v2s_bn.size(0), v2s_bn.size(1), vertices.size(2))
    v3s = vertices.index_select(1, v3s_bn.view(-1)).view(vertices.size(0) * v3s_bn.size(0), v3s_bn.size(1), vertices.size(2))

    # Create an index tensor
    indices = (torch.arange(0, vertices.size(0)) * (1 + vertices.size(0))).type(torch.cuda.LongTensor)

    # Select rows from v0s, v1s, v2s, v3s using the indices tensor
    v0s = v0s.index_select(0, indices)
    v1s = v1s.index_select(0, indices)
    v2s = v2s.index_select(0, indices)
    v3s = v3s.index_select(0, indices)

    # Compute loss
    cb1, cb1_l1 = _compute_loss(v1s - v0s, v2s - v0s)
    cb2, cb2_l1 = _compute_loss(v1s - v0s, v3s - v0s)

    cos = torch.sum(cb1*cb2, dim=2) / (cb1_l1 * cb2_l1 + eps)
    loss = torch.sum((cos+1).pow(2)) / batch_size
    return loss


def get_smoothness_loss_stage1(vertices, parameters, eps=1e-6):
    """ 
        Define a laplacian coordinate for each vertex
    """ 
    v0s, v1s, v2s, v3s = [torch.from_numpy(param).type(torch.cuda.LongTensor) for param in parameters]
    v0s, v1s, v2s, v3s = vertices[:, v0s], vertices[:, v1s], vertices[:, v2s], vertices[:, v3s]
    batch_size = vertices.size(0)
    
    cb1, cb1_l1 = _compute_loss(v1s - v0s, v2s - v0s)
    cb2, cb2_l1 = _compute_loss(v1s - v0s, v3s - v0s)
    
    loss = torch.sum(((torch.sum(cb1 * cb2, dim=2) / (cb1_l1 * cb2_l1 + eps)) + 1).pow(2))
    loss /= batch_size
    return loss



# Normalization
def normalize(input, p=2, dim=1, eps=1e-12):
    """ Measures the normal consistency between the generated mesh and ground truth """ 
    input = input / input.norm(p, dim)
    input = input.clamp(min=eps)
    input = input.unsqueeze(dim).expand_as(input)
    return input


def get_normal_loss(vertices, faces, gt_normals, idx2):
    # Find edges in faces tensor
    edges = torch.cat((faces[:,:,:2], faces[:,:,[0,2]], faces[:,:,1:]), 1)
    
    # Select vertices corresponding to edges
    edges_vertices = vertices.index_select(1, edges.view(-1)).view(vertices.size(0) * edges.size(0), edges.size(1), edges.size(2), vertices.size(2))
    
    # Calculate length of edges
    edges_len1 = edges_vertices[:,:,0] - edges_vertices[:,:,1]
    edges_len2 = edges_vertices[:,:,1] - edges_vertices[:,:,0]
    edges_vector = torch.stack((edges_len1, edges_len2), 2)
    
    # Select ground truth normals corresponding to edges
    gt_normals = gt_normals.index_select(1, idx2.contiguous().view(-1)).contiguous().view(gt_normals.size(0) * idx2.size(0), idx2.size(1), gt_normals.size(2))
    gt_normals_edges = gt_normals.index_select(1, edges.view(-1)).view(gt_normals.size(0) * edges.size(0), edges.size(1), edges.size(2), gt_normals.size(2))
    
    # Normalize edge vector and ground truth normal tensors
    gt_normals_edges = normalize(gt_normals_edges, p=2, dim=3)
    edges_vector = normalize(edges_vector, p=2, dim=3)
    
    # Calculate cosine between edge vector and ground truth normal
    cos = torch.abs(torch.sum(torch.mul(edges_vector, gt_normals_edges), 3))
    
    # Calculate mean normal loss
    nonzero = len(cos.nonzero())
    normal_loss = torch.sum(cos) / nonzero
    return normal_loss


# Edge 
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


# MSE
 
def calculate_l2_loss(x, y):
    """ L2 loss """ 
    l2 = nn.MSELoss() 
    return l2(x,y)

