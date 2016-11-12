import numpy as np
from numpy.linalg import eig, norm

import math
import nibabel as nib
#np.set_printoptions(threshold=np.inf)

# Input: 4D array of n brain scans
def create_eigenbrains(blocks, nii):
    n_blocks = blocks.shape[3]
    matrix = flatten_blocks(blocks, n_blocks)
    
    print("NII img shape: " + str(blocks.shape))
    print("Number of blocks: %d" % (n_blocks))
    print("Flattened matrix shape: " + str(matrix.shape))
    
    normalized_mtx = mean_std_normalize(matrix)
    normalized_mtx = remove_nan(normalized_mtx)
    
    # Z^t by Z covariance matrix, but will be huge for MRI data (# voxels in a block, squared)
    #covariance_mtx = np.true_divide(np.dot(normalized_mtx.transpose(), normalized_mtx), n_blocks-1)
    # Covariance matrix Zhang et al. used, smaller size (n_blocks squared)
    covariance_mtx = np.true_divide(np.dot(normalized_mtx, normalized_mtx.transpose()), n_blocks-1)
    
    eig_vals, eig_vectors = eig(covariance_mtx)
    eig_vals, eig_vectors = sort_eig_vectors(eig_vals,eig_vectors)
    
    eig_brains = np.dot(eig_vectors, normalized_mtx)
    #eig_brains = normalize_vectors(eig_brains)
    eig_brains = unflatten_blocks(eig_brains, blocks.shape)
    eig_brains = eig_brains.astype('float32')
    
    print("Eigenbrains shape: " + str(eig_brains.shape))
    
    nii_img = nib.Nifti1Image(eig_brains, nii.affine)
    nib.save(nii_img, "eigenbrains.nii.gz")
    print("Eigenbrains saved to eigenbrains.nii.gz")


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
    
def mean_std_normalize(matrix):
    col_means = np.mean(matrix, axis=0)
    col_std_devs = np.std(matrix, axis=0)
    normalized_mtx = np.true_divide(np.subtract(matrix, col_means), col_std_devs)
    return normalized_mtx 
    
def scale_to_unit_vector(v):
    norm_v = norm(v)
    if norm_v == 0:
        return v
    return np.true_divide(v,norm_v)

def normalize_vectors(mtx):
    for v in mtx:
        v = scale_to_unit_vector(v)
    return mtx

    


path = "../test/filtered_func_data.nii.gz"
#path = "../test/newsirp_final_XML.nii"
nii = nib.load(path)
brains = nii.get_data()

create_eigenbrains(brains, nii)

