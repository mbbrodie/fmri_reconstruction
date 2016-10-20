import numpy as np

def gen_img_set(size):
    img_set = []
    for x in range(0,size):
        img_set.append(gen_rand_img())
    return np.asarray(img_set)

def gen_rand_img():
    image = np.ndarray((10,10), dtype=np.int)
    for row in image:
        for x in range(0,10):
            row[x] = random.randint(0,1)
    return image
