import numpy as np
from numpy.linalg import eig, norm

import math
import nibabel as nib
#np.set_printoptions(threshold=np.inf)

# Input: 4D array of n brain scans
def create_eigenbrains(blocks, nii):
    n_blocks = blocks.shape[0]
    matrix = flatten_blocks(blocks)
    
    normalized_mtx = mean_std_normalize(matrix)
    normalized_mtx = remove_nan(normalized_mtx)
    
    # Z^t by Z covariance matrix, but will be huge for MRI data (# voxels in a block, squared)
    #covariance_mtx = np.true_divide(np.dot(normalized_mtx.transpose(), normalized_mtx), n_blocks-1)
    # Covariance matrix Zhang et al. used, smaller size (n_blocks squared)
    covariance_mtx = np.true_divide(np.dot(normalized_mtx, normalized_mtx.transpose()), n_blocks-1)
    
    eig_vals, eig_vectors = eig(covariance_mtx)
    eig_vals, eig_vectors = sort_eig_vectors(eig_vals,eig_vectors)
    
    eig_brains = np.dot(eig_vectors, normalized_mtx)
    eig_brains = normalize_vectors(eig_brains)
    
    eig_brains = eig_brains.reshape(blocks.shape,order='F')
    eig_brains = eig_brains.astype('float32')
    
    print(blocks.shape)
    print(eig_brains[0].shape)
    print(eig_brains.shape)
    
    nii_img = nib.Nifti1Image(eig_brains, nii.affine)
    nib.save(nii_img, "eigenbrains.nii.gz")
    
    
    # Glass brain visualization. Looks worse
    nii_small = nib.Nifti1Image(eig_brains[0], nii.affine)
    from nilearn import plotting
    plotting.plot_glass_brain(nii_small)
    plotting.show()
   
    
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
    
def flatten_blocks(blocks):
    matrix = []
    for block in blocks:
        matrix.append(block.flatten())
    matrix = np.asarray(matrix)
    return matrix
    
    
def scale_to_unit_vector(v):
    norm_v = norm(v)
    if norm_v == 0: 
       return v
    return np.true_divide(v,norm_v)

def normalize_vectors(mtx):
    for v in mtx:
        v = scale_to_unit_vector(v)
    return mtx

    
        

    
#brains = np.asarray([[[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]],[[[0,0,0],[0,0,0],[0,0,0]],[[2,2,2],[2,2,2],[2,2,2]], [[2,2,2],[2,2,2],[2,2,2]]],[[[2,2,2],[2,2,2],[2,2,2]],[[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[0,0,0],[0,0,0]]],[[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]]])

#path = "../test/filtered_func_data.nii.gz"
path = "../test/newsirp_final_XML.nii"
nii = nib.load(path)
brains = nii.get_data()
#print(nii)

print("\n\n")
create_eigenbrains(brains, nii)

