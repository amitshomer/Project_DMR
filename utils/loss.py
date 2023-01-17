import torch
from torch import nn
import numpy as np


# Smoomthness 
def get_v0v1_faces(faces, v0, v1):
    """ 
        Find faces that contain v0, v1 in this order
    """
    mask1 = np.logical_and(faces[:, 0] == v0, faces[:, 1] == v1)
    mask2 = np.logical_and(faces[:, 1] == v0, faces[:, 2] == v1)
    mask3 = np.logical_and(faces[:, 0] == v0, faces[:, 2] == v1)
    return np.logical_or(mask1, np.logical_or(mask2, mask3))


def smoothness_loss_parameters(faces):
    """
        Returns a list of 4 neighbours for each vertex 
        faces - shape [b, 5120, 3]
    """
    if hasattr(faces, 'get'): faces = faces.get()
    
    # Extract edge vertices from faces
    edge1, edge2 = faces[:, 0:2], faces[:, 1:3]
    edges12 = list(set([tuple(v) for v in np.sort(np.concatenate((edge1, edge2), axis=0))]))
    
    # v0s and v1s are the first and the second vertices that define an edge 
    v0s, v1s = np.array([v[0] for v in edges12], 'int32'), np.array([v[1] for v in edges12], 'int32')
    v2s, v3s = [], []
    
    # Iterate over each edge 
    for v0, v1 in zip(v0s, v1s):
        # Find faces that contain this edge
        mask1 = get_v0v1_faces(faces, v0, v1)
        mask2 = get_v0v1_faces(faces, v1, v0)
        mask = np.logical_or(mask1, mask2)
        face_vertices = faces[mask]
        
        # Check if there are more than one face containing this edge
        if face_vertices.shape[0] > 1:
            # Extract first neighbour vertex from first face
            v2s.append(int(np.setdiff1d(face_vertices[0], [v0, v1])[0]))
            # Extract second neighbour vertex from second face
            v3s.append(int(np.setdiff1d(face_vertices[1], [v0, v1])[0]))
        else:
            # Extract first neighbour vertex from first face
            v2s.append(int(np.setdiff1d(face_vertices[0], [v0, v1])[0]))
            # Append 0 to v3s if this edge included in one face only, no neighbour
            v3s.append(0)
    
    v2s, v3s = np.array(v2s, 'int32'), np.array(v3s, 'int32')
    return v0s, v1s, v2s, v3s


def get_weights(vec1, vec2, eps=1e-6):
    # Calculate the L2 norm of vec1 and vec2
    vec1_L2 = torch.sum(vec1.pow(2),dim=2)
    vec2_L2 = torch.sum(vec2.pow(2),dim=2)

    # Calculate the L1 norm of vec1 and vec2
    vec1_L1 = torch.sqrt(vec1_L2 + eps)
    vec2_L1 = torch.sqrt(vec2_L2 + eps)

    # Calculate the inner product of vec1 and vec2
    inner_product = (vec1 * vec2).sum(dim=2)
    
    # Calculate the cosine & sine of the angle between vec1 and vec2
    cos_angle = inner_product / (vec1_L1 * vec2_L1 + eps)
    sin_angle = torch.sqrt(1 - cos_angle.pow(2) + eps)

    # Finding an orthonormal basis  
    w = vec2 - vec1 * (((inner_product / (vec1_L2 + eps)).unsqueeze(2)).expand_as(vec1))
    w_L1 = vec2_L1 * sin_angle

    return w, w_L1


def get_neighbours_coords(vs, faces_bn,vertices):
    ''' Was taken from the original code'''
    
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
    v0s_bn, v1s_bn, v2s_bn, v3s_bn = final[:,:,0], final[:,:,1], final[:,:,2], final[:,:,3]

    # Reshape the vertices tensor and select the rows using the indices
    v0s = vertices.index_select(1, v0s_bn.view(-1)).view(vertices.size(0) * v0s_bn.size(0), v0s_bn.size(1), vertices.size(2))
    v1s = vertices.index_select(1, v1s_bn.view(-1)).view(vertices.size(0) * v1s_bn.size(0), v1s_bn.size(1), vertices.size(2))
    v2s = vertices.index_select(1, v2s_bn.view(-1)).view(vertices.size(0) * v2s_bn.size(0), v2s_bn.size(1), vertices.size(2))
    v3s = vertices.index_select(1, v3s_bn.view(-1)).view(vertices.size(0) * v3s_bn.size(0), v3s_bn.size(1), vertices.size(2))

    # Create an index tensor
    indices = (torch.arange(0, vertices.size(0)) * (1 + vertices.size(0))).type(torch.cuda.LongTensor)

    # Select rows from v0s, v1s, v2s, v3s using the indices tensor
    v0s, v1s, v2s, v3s = v0s.index_select(0, indices), v1s.index_select(0, indices),\
                            v2s.index_select(0, indices), v3s.index_select(0, indices)
    return v0s, v1s, v2s, v3s 


def get_smoothness_loss(vertices, parameters, faces_bn, eps=1e-6):
    ## Partially baed on the offical Repo of the paper 
    # Convert params to tensors
    batch_size = vertices.size(0)
    v0s, v1s, v2s, v3s = [torch.from_numpy(param).type(torch.cuda.LongTensor) for param in parameters]
    vs = torch.stack((v0s, v1s, v2s, v3s), 1)

    v0s, v1s, v2s, v3s = get_neighbours_coords(vs, faces_bn, vertices)

    # Get orthonormal basis
    w1, w1_L1 = get_weights(v1s - v0s, v2s - v0s)
    w2, w2_L1 = get_weights(v1s - v0s, v3s - v0s)

    loss = torch.sum(((torch.sum(w1 * w2, dim=2) / (w1_L1 * w2_L1 + eps)) + 1).pow(2))
    loss /= batch_size
    return loss


def get_smoothness_loss_stage1(vertices, parameters, eps=1e-6):
    ## Partially baed on the offical Repo of the paper 
    batch_size = vertices.size(0)
    v0s, v1s, v2s, v3s = [torch.from_numpy(param).type(torch.cuda.LongTensor) for param in parameters]
    v0s, v1s, v2s, v3s = vertices[:, v0s], vertices[:, v1s], vertices[:, v2s], vertices[:, v3s]
    
    # Get weights
    w1, w1_L1 = get_weights(v1s - v0s, v2s - v0s)
    w2, w2_L1 = get_weights(v1s - v0s, v3s - v0s)

    loss = torch.sum(((torch.sum(w1 * w2, dim=2) / (w1_L1 * w2_L1 + eps)) + 1).pow(2))
    loss /= batch_size
    return loss



# Normalization
def normalize_vector(v, p=2, dim=1, eps=1e-12):
    """ 
        Normalize vector
    """ 
    v_size = v.norm(p, dim).clamp(min=eps).unsqueeze(dim).expand_as(v)
    v = v / v_size
    return v


def inner_product(v1, v2):
    """ 
        Calc the inner product for a given two vectors - (v1/||v1|| * v2/||v2||)
    """
    # Normalize each vector
    v1 = normalize_vector(v1, p=2, dim=3)
    v2 = normalize_vector(v2, p=2, dim=3)
    
    # Inner product of the two normalized vectors
    cos = torch.sum(torch.mul(v1, v2), 3)

    # For pocisitve loss
    cos = torch.abs(cos) 
    return cos


def get_normal_loss(vertices, faces, gt_normals, q):
    """ 
        Requires the edge between a vertex with its neighbors to perpendicular (cos(thete)=0) to the observation from the ground truth
        q - for each vertex, its closest vertex that is found when calculating the chamfer loss. Shape [b, num_of_vertices]
        gt_normal - is the observed surface normal from ground truth. Shape - [b, faces_samples, 3_coords]

    """
    q = q.type(torch.cuda.LongTensor).detach()

    # Find edges in faces tensor & Get vertices coords for each edge - [2, 15360, 2, 3]
    edges = torch.cat((faces[:,:,:2], faces[:,:,[0,2]], faces[:,:,1:]), 1)
    edges_vertices_coords, indices = get_faces_edge_coords(vertices, edges)
    
    # Calculate third edge length in triangler, both directions 
    edge_ps = edges_vertices_coords[:,:,0] - edges_vertices_coords[:,:,1]
    edge_ng = edges_vertices_coords[:,:,1] - edges_vertices_coords[:,:,0]
    edges_vector = torch.stack((edge_ps, edge_ng), 2) 
    
    # Ground truth normals for the face that contains the closest vertex q 
    gt_normals = gt_normals.index_select(1, q.contiguous().view(-1)).contiguous().view(gt_normals.size(0) * q.size(0), q.size(1), gt_normals.size(2))
    gt_normals = gt_normals.index_select(0, indices) # [b, num_of_vertices, 3_coords]
    
    # Ground truth edges
    gt_normals_edges = gt_normals.index_select(1, edges.view(-1)).view(gt_normals.size(0) * edges.size(0), edges.size(1), edges.size(2), gt_normals.size(2))
    gt_normals_edges = gt_normals_edges.index_select(0, indices) #[2, 15360, 2, 3])
    
    # Avg angle between gt normal and deformed face
    cos = inner_product(edges_vector, gt_normals_edges)
    nonzero = len(cos.nonzero())
    normal_loss = torch.sum(cos) / nonzero
    return normal_loss


# Edge 
def get_faces_edge_coords(vertices, edges):
    bs_edges, edges_num, edges_vertices_num = edges.size(0),edges.size(1),edges.size(2) # [b, 3*5120, 2-edge1,edge2]
    bs_vertices, vertex_coors_num = vertices.size(0), vertices.size(2)  # b, 3 (x,y,z)
    
    # Get vertices coords for each edge
    edge_coords = vertices.index_select(1,edges.view(-1)).\
        view(bs_vertices*bs_edges, edges_num, edges_vertices_num, vertex_coors_num) # [b*b, 3*5120, 2, 3] 
    indices = (torch.arange(0,bs_vertices)*(1+bs_vertices)).type(torch.cuda.LongTensor)
    edge_coords = edge_coords.index_select(0,indices) # [b, 3*5120, 2, 3]

    return edge_coords, indices


def get_avg_edge(edges_vertices):
    """ Returns the avg edge length without taking into account zero values"""
    # Edge lenght
    edges_len = torch.norm((edges_vertices[:,:,0]-edges_vertices[:,:,1]), 2, 2)
    edges_len = torch.pow(edges_len, 2)

    # Edge total normalize sum
    nonzero = len(edges_len.nonzero()) # amount of non-zero elements
    edge_loss = torch.sum(edges_len)/nonzero
    return edge_loss


def get_edge_loss_stage1(vertices,edge): 
    vertices_edge = vertices.index_select(1,edge.view(-1)).\
        view(vertices.size(0),edge.size(0),edge.size(1),vertices.size(2))
    vertices_edge_vector = vertices_edge[:,:,0] - vertices_edge[:,:,1]
    vertices_edge_len = torch.pow(vertices_edge_vector.norm(2,2),2)
    edge_loss = torch.mean(vertices_edge_len)
    return edge_loss


def get_edge_loss(vertices, x, stage=None):
    """  
        Penalizes flying vertices, which ususally cause long edges
        vertices - shape [b, 2562, 3]
        x - depending on the stage: 
            if stage=1, x is the edges shape [b, 3*5120, 2]
            else, x is the faces shape [b, 5120, 3]
    """ 
    if (stage == 1): 
        edges = x
    else: 
        faces = x
        edge1, edge2, edge3 = faces[:,:,:2], faces[:,:,[0,2]] ,faces[:,:,1:]
        edges = torch.cat((edge1, edge2, edge3),1) # [b, 3*5120, 2]

    # Get vertices coords for each edge
    edges_vertices_coords, _ = get_faces_edge_coords(vertices, edges)

    # Get average edge leght
    edge_loss = get_avg_edge(edges_vertices_coords)

    return edge_loss


# MSE
 
def mse_loss(x, y):
    """ L2 loss """ 
    l2 = nn.MSELoss() 
    return l2(x,y)

