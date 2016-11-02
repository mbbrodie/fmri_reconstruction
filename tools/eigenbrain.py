import numpy as np
from scipy.linalg import eig

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
    #eig_brains = eig_vectors.reshape(blocks.shape)
    
    print(eig_vectors.shape)
    
        

    
brains = np.asarray([[[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]],[[[0,0,0],[0,0,0],[0,0,0]],[[2,2,2],[2,2,2],[2,2,2]], [[2,2,2],[2,2,2],[2,2,2]]],[[[2,2,2],[2,2,2],[2,2,2]],[[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[0,0,0],[0,0,144]]],[[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]]])
print(brains)
print("\n\n")
create_eigenbrains(brains)

