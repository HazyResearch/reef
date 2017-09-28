import numpy as np

class DataLoader(object):
    """ A class to load in appropriate numpy arrays
    """

    def load_data(self, dataset, data_path='data/', ):
        if dataset == 'bone_tumor':
            primitive_matrix = np.load(data_path+dataset+'/primitive_matrix.npy')
            ground = np.load(data_path+dataset+'/ground.npy')
            return primitive_matrix, ground
        elif dataset == 'mammogram':
            train_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_train.npy')
            val_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_val.npy')
            test_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_test.npy')

            train_ground = np.load(data_path+dataset+'/ground_train.npy')
            val_ground = np.load(data_path+dataset+'/ground_val.npy')
            test_ground = np.load(data_path+dataset+'/ground_test.npy')

            return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground, val_ground, test_ground



        
    