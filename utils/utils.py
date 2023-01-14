import numpy as np
import torch
import pandas as pd
from scipy.sparse import coo_matrix
import scipy 
from loss import get_v0v1_faces

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class AverageValueMeter:
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


def get_max_try(errors, sampled_face_index, fn=5120):
    """ 
        Extracts for each face the errors and indices of the sampled points belonging to that face. 
        Returns the maximum error for each face among the sampled points belonging to that face.
        
        NOTE This implementation acheived the same results, however it is much slower than the original, therefore, not used.. 
    """
    batch_size = errors.shape[0]
    max_errors = []
    # For each image in batch
    for i in range(batch_size):
        face_errors = errors[i, :]
        face_index = sampled_face_index[i]

        # Get unique sampled faces indices 
        uni = torch.unique(face_index)

        # Find relevant errors for each sampled face
        uni_faces_ind = (uni.unsqueeze(1)).repeat(1,face_errors.shape[0])
        faces_index_repeat = (face_index.unsqueeze(1)).repeat(1,uni_faces_ind.shape[0]).T
        faces_error_repeat = (face_errors.unsqueeze(1).T).repeat(uni_faces_ind.shape[0],1)
        mask = faces_index_repeat == uni_faces_ind

        # Find max for each sampled face
        sf_max_error = torch.zeros_like(faces_error_repeat)
        sf_max_error[mask] = faces_error_repeat[mask]
        sf_max_error = sf_max_error.max(1)[0] # Sampled faces max error
        
        all_faces_errors = torch.zeros(fn)
        all_faces_errors[uni]=sf_max_error
        max_errors.append(all_faces_errors)
    
    max_errors = torch.stack(max_errors)
    return max_errors


def get_max(errors, index, fn=5120): 
    # Was implemented - see get_max_try - acheived same results however a bit slower. 
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
    return max_errors[:,:,0]


def get_edges(faces):
    ''' Return unique edge list'''
    edge = []
    for i, j in enumerate(faces):
        edge.append(j[:2])
        edge.append(j[1:])
        edge.append(j[[0, 2]])
    edge = np.array(edge)

    # Represent each edge with a scalar
    edge_im = edge[:, 0] * edge[:, 1] + (edge[:, 0] + edge[:, 1]) * 1j
    
    # Remove repetitions (unique scalars)
    edge_unique = edge[ np.unique(edge_im, return_index=True)[1]]
    
    edge_cuda = (torch.from_numpy(edge_unique).type(torch.cuda.LongTensor)).detach()
    return edge_cuda


def find_boundary_edges_try(triangles):
    # Extract edge vertices from faces
    edges = np.concatenate((triangles[:, :2], triangles[:, [0, 2]], triangles[:, 1:]), 0) # array of edges
    edges.sort(1)
    
    # v0s and v1s are the first and the second vertices that define an edge 
    v0s, v1s = np.array([v[0] for v in edges], 'int32'), np.array([v[1] for v in edges], 'int32')
    boundary_edge = []

    triangles_copy = np.copy(triangles)
    # Iterate over each edge 
    for v0, v1 in zip(v0s, v1s):
        # Find faces that contain this edge
        mask1 = get_v0v1_faces(triangles_copy, v0, v1)
        mask2 = get_v0v1_faces(triangles_copy, v1, v0)
        mask = np.logical_or(mask1, mask2)
        
        face_vertices = triangles_copy[mask]
        
        if face_vertices.shape[0] == 1:
            boundary_edge.append((v0, v1))
    
    return np.array(boundary_edge)


def get_boundary_try(faces):
    ''' 
        Boundary edges are not shared between faces, meaning there's only one face that include this edge
        NOTE This implementation acheived the same results however it is much slower than the original, therefore, not used.. 
    '''
    triangles = faces.cpu().data.numpy()
    triangles = triangles[triangles.sum(1).nonzero()]

    # Get boundary edges
    boundary_edge = find_boundary_edges_try(triangles)
    boundary_edge = np.array(sorted(boundary_edge, key=lambda x: (x[0], x[1])))

    # For each boundary vertex, return its two edges 
    boundary_edge_inverse = boundary_edge[:, [1, 0]]
    boundary_edge_all = np.concatenate((boundary_edge, boundary_edge_inverse), 0)
    boundary_edge_all = boundary_edge_all[np.argsort(boundary_edge_all[:, 0])]
    
    # Vertices 
    boundary_vertices = np.unique(boundary_edge)
    boundary_vertices = torch.from_numpy(boundary_vertices).type(torch.cuda.LongTensor)

    # Return boundary vertices  
    selected_point = np.where(boundary_edge_all[:, 0] == np.concatenate((boundary_edge_all[1:, 0],
                                                                         boundary_edge_all[:1, 0]), 0))
    boundary_pair = np.concatenate((boundary_edge_all[selected_point[0]],
                                    boundary_edge_all[selected_point[0] + 1][:, 1:]), 1)
    boundary_pair = torch.from_numpy(boundary_pair).type(torch.cuda.LongTensor)
    
    return boundary_pair, boundary_vertices, boundary_edge


def get_boundary(faces):
    ''' 
        Original Implementation. Boundary edges are not shared between faces, meaning there's only one face that include this edge
        NOTE alternative implementation (get_boundary_try) is too slow, using the original
    '''

    # Face = v1, v2, v3
    vertices_num = faces.max().item() + 1
    faces_np = faces.cpu().data.numpy()
    faces_np = faces_np[faces_np.sum(1).nonzero()] # Remove Zeros - pruned faces

    # All Edges - each faces and its edges, shared edges listed twice 
    edge1 = faces_np[:, :2] # v2-->v1 
    edge2 = faces_np[:, [0, 2]] # v3-->v1
    edge3 = faces_np[:, 1:] # v3-->v2
    face_edges = np.concatenate((edge1, edge2, edge3), 0)
    face_edges.sort(1)

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

    boundary_vertices = np.unique(np.concatenate((boundary_V1, boundary_V2), 0))
    boundary_vertices = torch.from_numpy(boundary_vertices).type(torch.cuda.LongTensor)

    # For each boundary vertex, return its two edges 
    boundary_edge_inverse = boundary_edge[:, [1, 0]]
    boundary_edge_all = np.concatenate((boundary_edge, boundary_edge_inverse), 0)
    boundary_edge_all = boundary_edge_all[np.argsort(boundary_edge_all[:, 0])]
    
    # Return boundary vertices  
    selected_point = np.where(boundary_edge_all[:, 0] == np.concatenate((boundary_edge_all[1:, 0],
                                                                         boundary_edge_all[:1, 0]), 0))
    boundary_pair = np.concatenate((boundary_edge_all[selected_point[0]],
                                    boundary_edge_all[selected_point[0] + 1][:, 1:]), 1)
    boundary_pair = torch.from_numpy(boundary_pair).type(torch.cuda.LongTensor)
    
    return boundary_pair, boundary_vertices, boundary_edge


def faces_selection(sampling_number, device, faces_points):
    ''' Select faces to be sampled by normal size ''' 

    faces = faces_points.cpu().data.numpy() # [b, n_faces, 3 vertices, 3 coords]
    v1, v2, v3 = faces[:, :, 0], faces[:, :, 1], faces[:, :, 2] # [b, n_faces, 3 coords]
    
    # Calculate cross-product triangles's vectors
    normal = np.cross(v2 - v1, v3 - v1) 
    # Calculate (x^2 + y^2 + z^2)^0.5 for each face
    normal_size = np.sqrt(np.sum(normal ** 2, axis=-1)) # [b, n_faces]
    # Calculate the sum of normals for each batch
    normal_sum = np.sum(normal_size, axis=1) # [b]
    # Calculate the cumulative sum of normals for each batch
    normal_cum = np.cumsum(normal_size, axis=1) # [b, n_faces]
    # Generate random numbers between 0 and 1
    faces_pick = normal_sum[:, np.newaxis] * np.random.random(sampling_number)[np.newaxis, :]

    # Select faces using the cumulative sum of normals
    faces_index = []
    for i in range(normal_cum.shape[0]):
        # Find the index of the selected faces using the cumulative sum of normals
        index = np.searchsorted(normal_cum[i], faces_pick[i])
        # Append the selected indices to the list
        faces_index.append(index)

    # Clip the values to the range [0, n_faces - 1]
    faces_index = np.clip(np.array(faces_index), 0, normal_cum.shape[1] - 1)
    faces_index_tensor = torch.from_numpy(faces_index).to(device).type(torch.cuda.LongTensor).to(device)
    
    # Sort the indices in ascending order
    faces_index_tensor_sort = faces_index_tensor.sort(1)[0]
    return faces_index_tensor_sort


def faces_selection_try(samples_num, device, faces_points):
     ''' Select faces to be sampled ''' 

     faces = faces_points.cpu().data.numpy() # [b, n_faces, 3 vertices, 3 coords]
     random_face_ind = np.sort(np.round(np.random.rand(faces.shape[0], samples_num) * faces.shape[1]), axis=1).astype(int) # rand indx in range (0,1) * num_face
     faces_index_tensor = (torch.from_numpy(random_face_ind).to(device)).type(torch.cuda.LongTensor)

     return faces_index_tensor


def samples_random(faces_cuda, pointsRec, sampled_number,device='cuda:0'): 
    """ Was not implemented. Random vertices sampleing on given faces """ 

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

    # Select the vertices of the sampled triangles
    tri_origins = faces_points[:, :, 0].clone()
    tri_vectors = faces_points[:, :, 1:].clone()
    tri_vectors = tri_vectors - tri_origins.unsqueeze(2).expand_as(tri_vectors)

    tri_origins = tri_origins.index_select(1, faces_index_sample.view(-1)).view(
        tri_origins.size()[0] * faces_index_sample.size()[0],
        faces_index_sample.size()[1], tri_origins.size()[2])
    tri_vectors = tri_vectors.index_select(1, faces_index_sample.view(-1)).view(
        tri_vectors.size()[0] * faces_index_sample.size()[0],
        faces_index_sample.size()[1], tri_vectors.size()[2], tri_vectors.size()[3])

    diag_index = (torch.arange(0, pointsRec.size()[0]).to(device))
    diag_index=diag_index.type(torch.cuda.LongTensor)
    diag_index = (1+faces_index_sample.size(0)) * diag_index

    tri_origins = tri_origins.index_select(0, diag_index)
    tri_vectors = tri_vectors.index_select(0, diag_index)

    random_lenghts = ((torch.randn(pointsRec.size()[0], tri_origins.size()[1], 2, 1).uniform_(0, 1)).to(device))
    random_test = random_lenghts.sum(2).squeeze(2) > 1.0
    random_test_minus = random_test.type(torch.cuda.FloatTensor).unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 1)
    random_lenghts = torch.abs(random_lenghts - random_test_minus)
    random_lenghts = random_lenghts.repeat(1,1,1,3)

    sample_vector = (tri_vectors * random_lenghts).sum(2)
    samples = sample_vector + tri_origins

    return samples, faces_index_sample


def prune(faces_cuda_bn, error, tau, index, pool='max', faces_number=5120, device='cuda:0'):
    ''' Remove bounday faces with error larger than tau'''
    # Decrease tau by a constant factor
    tau = tau / 10.0 

    # Positive error
    error = torch.pow(error, 2)

    # For each face, get the max error of its samples
    face_error = get_max(error.cpu(), index.cpu(), faces_number).to(device)

    # Mark faces to prune as zero
    faces_cuda_bn = faces_cuda_bn.clone()
    faces_cuda_bn[face_error > tau] = 0 # [0,0,0] - pruned face

    faces_cuda_set = []
    for k in torch.arange(0, error.size(0)):
        faces_cuda = faces_cuda_bn[k]
        
        # Get boundary points
        _, _, boundary_edge = get_boundary(faces_cuda)
        boundary_edge_point = boundary_edge.astype(np.int64).reshape(-1)

        # 
        counts = pd.value_counts(boundary_edge_point)
        toremove_point = torch.from_numpy(np.array(counts[counts > 2].index)).to(device)
        faces_cuda_expand = faces_cuda.unsqueeze(2).expand(faces_cuda.shape[0], faces_cuda.shape[1],
                                                           toremove_point.shape[0])
        toremove_point_expand = toremove_point.unsqueeze(0).unsqueeze(0).\
            expand(faces_cuda.shape[0],faces_cuda.shape[1],toremove_point.shape[0])
        toremove_index = ((toremove_point_expand == faces_cuda_expand).sum(2).sum(1)) != 0
        faces_cuda[toremove_index] = 0
        triangles = faces_cuda.cpu().data.numpy()

        # Removing Single Occurrence vertex in faces
        v = pd.value_counts(triangles.reshape(-1))
        v = v[v == 1].index
        for vi in v:
            if np.argwhere(triangles == vi).shape[0] == 0:
                continue
            triangles[np.argwhere(triangles == vi)[0][0]] = 0

        # Append pruned triangles to list
        faces_cuda_set.append(torch.from_numpy(triangles).to(device).unsqueeze(0))

    # Triangles after prune
    faces_cuda_bn = torch.cat(faces_cuda_set, 0)
    return faces_cuda_bn


def get_boundary_points_bn(faces_cuda_bn, pointsRec_refined, device='cuda:0'):
    " Was not implemented"
    selected_pair_all, selected_pair_all_len, boundary_points_all, boundary_points_all_len = [], [], [], []

    for edge_bn in torch.arange(0, faces_cuda_bn.shape[0]):
        faces_each = faces_cuda_bn[edge_bn]
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


def create_round_spehere(num_vertices, cuda = 'cuda:0'):
    name = 'sphere' + str(num_vertices) + '.mat'
    mesh = scipy.io.loadmat('./data/' + name)
    faces = np.array(mesh['f'])
    faces_cuda = torch.from_numpy(faces.astype(int)).type(torch.cuda.LongTensor).to(cuda)
    vertices_sphere = np.array(mesh['v'])
    vertices_sphere = (torch.cuda.FloatTensor(vertices_sphere)).transpose(0, 1).contiguous()
    vertices_sphere = vertices_sphere.contiguous().unsqueeze(0).to(cuda)
    edge_cuda = get_edges(faces)
    return edge_cuda, vertices_sphere, faces_cuda , faces


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
