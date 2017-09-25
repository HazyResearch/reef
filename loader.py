import numpy as np

class DataLoader(object):
    """ A class to load in appropriate numpy arrays
    """

    def load_data(self, dataset, data_path='data/', ):
        primitive_matrix = np.load(data_path+dataset+'/primitive_matrix.npy')
        ground = np.load(data_path+dataset+'/ground.npy')
        return primitive_matrix, ground


        
    