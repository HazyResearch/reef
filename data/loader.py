import numpy as np
import scipy
from scipy import sparse

class DataLoader(object):
    """ A class to load in appropriate numpy arrays
    """

    def prune_features(self, val_primitive_matrix, train_primitive_matrix, thresh=0.01):
        val_sum = np.sum(np.abs(val_primitive_matrix),axis=0)
        train_sum = np.sum(np.abs(train_primitive_matrix),axis=0)

        #Only select the indices that fire more than 1% for both datasets
        train_idx = np.where((train_sum >= thresh*np.shape(train_primitive_matrix)[0]))[0]
        val_idx = np.where((val_sum >= thresh*np.shape(val_primitive_matrix)[0]))[0]
        common_idx = list(set(train_idx) & set(val_idx))

        return common_idx


    def load_data(self, dataset, data_path='/dfs/scratch0/paroma/data/'):
         #TODO: load all and split into train and test and validation here....
        if dataset == 'imdb':
            train_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_train.npy')
            train_primitive_matrix = np.array(train_primitive_matrix.item().todense())

            val_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_val.npy')
            val_primitive_matrix = np.array(val_primitive_matrix.item().todense())

            test_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_test.npy')
            test_primitive_matrix = np.array(test_primitive_matrix.item().todense())

            val_ground = np.load(data_path+dataset+'/ground_val.npy')
            train_ground = np.load(data_path+dataset+'/ground_train.npy')
            test_ground = np.load(data_path+dataset+'/ground_test.npy')

            common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
            return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix, train_ground, val_ground, test_ground