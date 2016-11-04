import numpy as np
from scipy.linalg import eig

import math
import nibabel as nib
#np.set_printoptions(threshold=np.inf)

# Input: 4D array of n brain scans
def create_eigenbrains(blocks, nii):
    matrix = []
    n_blocks = blocks.shape[0]
    for block in blocks:
        matrix.append(block.flatten())
    matrix = np.asarray(matrix)
    
    col_means = np.mean(matrix, axis=0)
    col_std_devs = np.std(matrix, axis=0)  
    normalized_mtx = np.true_divide(np.subtract(matrix, col_means), col_std_devs)
    
    nan = np.isnan(normalized_mtx)
    normalized_mtx[nan] = 0
    
    # Z^t by Z covariance matrix, but will be huge for MRI data (# voxels in a block, squared)
    #covariance_mtx = np.true_divide(np.dot(normalized_mtx.transpose(), normalized_mtx), n_blocks-1)
    # Covariance matrix Zhang et al. used, smaller size (n_blocks squared)
    covariance_mtx = np.true_divide(np.dot(normalized_mtx, normalized_mtx.transpose()), n_blocks-1)
    
    nan = np.isnan(covariance_mtx)
    covariance_mtx[nan] = 0
    
    eig_vals, eig_vectors = eig(covariance_mtx)

    sort_indexes = eig_vals.argsort()[::-1]   
    eig_vals = eig_vals[sort_indexes]
    eig_vectors = eig_vectors[:,sort_indexes]
    
    eig_brains = np.dot(eig_vectors, normalized_mtx)
    eig_brains = eig_brains.reshape(blocks.shape)
    eig_brains = eig_brains.astype('float32')
    
    print(blocks.shape)
    print(eig_brains[0].shape)
    print(eig_brains.shape)
    
    
    nii_img = nib.Nifti1Image(eig_brains, nii.affine)
    nib.save(nii_img, "eigenbrains.nii.gz")
    # Reshape eigenvectors into eigenbrains, (A*A matrix into A*x*y*z)
    #eig_brains = eig_vectors.reshape(eig_vectors.shape[0],blocks.shape[1],blocks.shape[2],blocks.shape[3])
    
    #print (eig_brains)

    
        

    
#brains = np.asarray([[[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]],[[[0,0,0],[0,0,0],[0,0,0]],[[2,2,2],[2,2,2],[2,2,2]], [[2,2,2],[2,2,2],[2,2,2]]],[[[2,2,2],[2,2,2],[2,2,2]],[[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[0,0,0],[0,0,0]]],[[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]]])

path = "filtered_func_data.nii.gz"
nii = nib.load(path)
brains = nii.get_data()
print(nii)

print("\n\n")
create_eigenbrains(brains, nii)

