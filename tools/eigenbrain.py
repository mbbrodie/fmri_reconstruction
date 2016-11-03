import numpy as np
from scipy.linalg import eig

import os
import nibabel as nib

# Input: 4D array of n brain scans
def create_eigenbrains(blocks):
    matrix = []
    n_blocks = blocks.shape[0]
    for block in blocks:
        matrix.append(block.flatten())
    matrix = np.asarray(matrix)
    
    col_means = np.mean(matrix, axis=0)
    col_std_devs = np.std(matrix, axis=0)  
    normalized_mtx = np.true_divide(np.subtract(matrix, col_means), col_std_devs)
    
    # True covariance matrix, but will be huge for MRI data (# voxels in a block, squared)
    covariance_mtx = np.true_divide(np.dot(normalized_mtx.transpose(), normalized_mtx), n_blocks-1)
    # Covariance matrix Zhang et al. used, smaller size (n_blocks squared)
    # covariance_mtx = np.true_divide(np.dot(normalized_mtx, normalized_mtx.transpose()), n_blocks-1)
    
    eig_vals, eig_vectors = eig(covariance_mtx)

    sort_indexes = eig_vals.argsort()[::-1]   
    eig_vals = eig_vals[sort_indexes]
    eig_vectors = eig_vectors[:,sort_indexes]
    
    # Reshape eigenvectors into eigenbrains, (A*A matrix into A*x*y*z)
    eig_brains = eig_vectors.reshape(eig_vectors.shape[0],blocks.shape[1],blocks.shape[2],blocks.shape[3])
    
    print (eig_brains)

    
        

    
#brains = np.asarray([[[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]],[[[0,0,0],[0,0,0],[0,0,0]],[[2,2,2],[2,2,2],[2,2,2]], [[2,2,2],[2,2,2],[2,2,2]]],[[[2,2,2],[2,2,2],[2,2,2]],[[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[0,0,0],[0,0,0]]],[[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]]])

path = "data_figure_run01.nii.gz"
brains = nib.load(path)
brains = brains.get_data()
print(brains.shape)

print("\n\n")
create_eigenbrains(brains)

