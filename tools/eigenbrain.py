import numpy as np
from numpy.linalg import eig, norm

import math
import nibabel as nib
#np.set_printoptions(threshold=np.inf)

# Input: list of paths to NII files, 4D array of n brain scans, example NII image
def create_eigenbrains(file_paths, blocks, nii):
    
    matrix, n_blocks, shape = load_and_flatten(file_paths)
    
    print("Number of blocks: %d" % (n_blocks))
    print("Flattened matrix shape: " + str(matrix.shape))
    
    normalized_mtx, col_means, col_std_devs = mean_std_normalize_matrix(matrix)
    normalized_mtx = remove_nan(normalized_mtx)
    
    # Z^t by Z covariance matrix, but will be huge for MRI data (# voxels in a block, squared)
    # covariance_mtx = np.true_divide(np.dot(normalized_mtx.transpose(), normalized_mtx), n_blocks-1)
    # Covariance matrix Zhang et al. used, smaller size (n_blocks squared)
    covariance_mtx = np.true_divide(np.dot(normalized_mtx, normalized_mtx.transpose()), n_blocks-1)
    
    eig_vals, eig_vectors = eig(covariance_mtx)
    eig_vals, eig_vectors = sort_eig_vectors(eig_vals,eig_vectors)
    
    eig_vectors = np.dot(eig_vectors, normalized_mtx)
    eig_vectors = normalize_vectors(eig_vectors)
    #print("Normalized Matrix Shape: " + str(normalized_mtx.shape))
    print(eig_vectors.shape)
    print (col_means.shape)
    
    return project_matrix_onto_eigspace(normalized_mtx, eig_vectors), eig_vectors, col_means, col_std_devs
    # eig_brains = unflatten_blocks(eig_brains, blocks.shape)
    # eig_brains = eig_brains.astype('float32')
    
    # print("Eigenbrains shape: " + str(eig_brains.shape))
    
    # nii_img = nib.Nifti1Image(eig_brains, nii.affine)
    # nib.save(nii_img, "/home/wesack/eigenbrains/eigenbrain_random_run20.nii.gz")
    # print("Eigenbrains saved to eigenbrains.nii.gz")


def flatten_blocks(blocks, n_blocks):
    matrix = []
    for x in range(n_blocks):
        block = blocks[:,:,:,x]
        matrix.append(block.flatten())
    return np.asarray(matrix)

def unflatten_blocks(blocks, shape):
    unflattened = np.empty(shape)
    for x in range(shape[3]):
        unflattened[:,:,:,x] = blocks[x].reshape(shape[0:3])
        
    unflattened = unflattened.astype('float32')
    return unflattened
    
def sort_eig_vectors(eig_vals, eig_vectors):
    sort_indexes = eig_vals.argsort()[::-1]   
    eig_vals = eig_vals[sort_indexes]
    eig_vectors = eig_vectors[:,sort_indexes]
    return eig_vals,eig_vectors
    
def remove_nan(mtx):
    nan = np.isnan(mtx)
    mtx[nan] = 0
    return mtx

def mean_std_normalize(matrix, col_means, col_std_devs):
    normalized_mtx = np.true_divide(np.subtract(matrix, col_means), col_std_devs)
    normalized_mtx = remove_nan(normalized_mtx)
    return normalized_mtx
    
def mean_std_normalize_matrix(matrix):
    col_means = np.mean(matrix, axis=0)
    col_std_devs = np.std(matrix, axis=0)
    normalized_mtx = np.true_divide(np.subtract(matrix, col_means), col_std_devs)
    normalized_mtx = remove_nan(normalized_mtx)
    return normalized_mtx, col_means, col_std_devs

def mean_std_normalize_vector(vector, col_means, col_std_devs):
    normalized_mtx = np.true_divide(np.subtract(matrix, col_means), col_std_devs)
    normalized_mtx = remove_nan(normalized_mtx)
    return normalized_mtx

def project_vector_onto_eigspace(vector, eig_vectors):
    projection = []
    for eig_vec in eig_vectors:
        weight = np.dot(eig_vec.transpose(), vector)
        projection.append(weight)
    return np.asarray(projection)

def project_matrix_onto_eigspace(matrix, eig_vectors):
    new_matrix = []
    for vector in matrix:
        new_matrix.append(project_vector_onto_eigspace(vector, eig_vectors))
    return np.asarray(new_matrix)    

def recreate_blocks(eigspace_blocks, eig_vectors, shape):
    blocks = []
    for eb in eigspace_blocks:
        sum_vec = None
        index = 0
        for eb_entry in eb:
            addition = np.multiply(eig_vectors[index], eb_entry)
            if sum_vec is None:
                sum_vec = addition
            else:
                sum_vec = np.add(sum_vec, addition)
            index = index + 1
        blocks.append(sum_vec)
    return np.asarray(blocks)

def recreate_blocks2(eb, eig_vectors, shape):
    blocks = []

    sum_vec = None
    index = 0
    for eb_entry in eb:
        addition = np.multiply(eig_vectors[index], eb_entry)
        if sum_vec is None:
            sum_vec = addition
        else:
            sum_vec = np.add(sum_vec, addition)
        index = index + 1
    blocks.append(sum_vec)
    return np.asarray(blocks)
    
def scale_to_unit_vector(v):
    norm_v = norm(v)
    if norm_v == 0:
        return v
    return np.true_divide(v,norm_v)

def normalize_vectors(mtx):
    for v in mtx:
        v = scale_to_unit_vector(v)
    return mtx

def load_and_flatten(file_paths):
    matrix = None
    n_blocks = 0
    
    for path in file_paths:
        my_nii = nib.load(path)
        scans = my_nii.get_data()
        shape = scans.shape
        block_count = scans.shape[3]
        n_blocks += block_count
        
        if matrix is None:
            matrix = flatten_blocks(scans, block_count)
        else:
            matrix = np.concatenate([matrix, flatten_blocks(scans, block_count)])
    return matrix, n_blocks, shape

def get_affine(paths):
    my_nii_path = paths[0]
    nii = nib.load(my_nii_path)
    return nii.affine

def save_nii(blocks, affine, filename):
    nii_img = nib.Nifti1Image(blocks, affine)
    nib.save(nii_img, filename)



    


#paths= ["/home/wesack/nilearn_data/miyawaki2008/func/data_figure_run09.nii.gz", "/home/wesack/nilearn_data/miyawaki2008/func/data_random_run09.nii.gz"]
#path = "../test/newsirp_final_XML.nii"
#nii = nib.load(path)
#brains = nii.get_data()

#create_eigenbrains(paths, None, None)

