import pickle

class Dataset:
    '''Contains helper functions for loading various fMRI datasets.
    Downloads and saves data if not already present in folder.
    Some functions return full dataset. Others return train/test sets.
    '''
    def __init__():
        pass

    def load_miyawaki_full():
        dataset = pickle.load(open('miyawaki_fixed_scaled.p','rb'))
        return dataset

    def load_miyawaki_masked():
        '''See http://nilearn.github.io/auto_examples/02_decoding/plot_miyawaki_reconstruction.html#sphx-glr-auto-examples-02-decoding-plot-miyawaki-reconstruction-py'''
