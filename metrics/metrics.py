import numpy as np

# Returns mean-squared error between two images
def calc_mse(img1, img2):
    sum = 0
    rows = 10
    cols = 10
    
    for x in range(0,rows):
        for y in range(0,cols):
            sum += (img1[x][y] - img2[x][y]) ** 2
    sum /= (rows * cols)
    return sum
