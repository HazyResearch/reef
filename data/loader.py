import numpy as np

class DataLoader(object):
    """ A class to load in appropriate numpy arrays
    """

    def load_data(self, dataset, data_path='data/', ):
         #TODO: load all and split into train and test and validation here....
        if dataset == 'bone_tumor':
            primitive_matrix = np.load(data_path+dataset+'/primitive_matrix.npy')
            ground = np.load(data_path+dataset+'/ground.npy')

            train_primitive_matrix = primitive_matrix[0:400,:]
            train_ground = ground[0:400]
            val_primitive_matrix = primitive_matrix[400:600,:]
            val_ground = ground[400:600]
            return train_primitive_matrix, val_primitive_matrix, [], train_ground, val_ground, []
       
        elif dataset == 'mammogram':
            train_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_train.npy')
            val_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_val.npy')
            test_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_test.npy')

            train_ground = np.load(data_path+dataset+'/ground_train.npy')
            val_ground = np.load(data_path+dataset+'/ground_val.npy')
            test_ground = np.load(data_path+dataset+'/ground_test.npy')

            return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground, val_ground, test_ground
        
        elif dataset == 'visual_genome':
            train_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_train.npy')
            val_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_val.npy')

            train_ground = np.load(data_path+dataset+'/ground_train.npy')
            val_ground = np.load(data_path+dataset+'/ground_val.npy')

            return train_primitive_matrix, val_primitive_matrix, [], train_ground, val_ground, []

        elif dataset == 'activity_net':
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=4)

            primitive_matrix = np.load(data_path+dataset+'/primitive_matrix.npy')
            ground = np.load(data_path+dataset+'/ground.npy')
            for train_index, test_index in kf.split(primitive_matrix):
                  val_primitive_matrix, train_primitive_matrix = primitive_matrix[train_index], primitive_matrix[test_index]
                  val_ground, train_ground = ground[train_index], ground[test_index]

            return train_primitive_matrix, val_primitive_matrix, [], train_ground, val_ground, []

        elif dataset == 'twitter':
            train_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_train.npy')
            train_primitive_matrix = np.array(train_primitive_matrix.item().todense())

            val_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_val.npy')
            val_primitive_matrix = np.array(val_primitive_matrix.item().todense())

            val_ground = np.load(data_path+dataset+'/ground_val.npy')

            return train_primitive_matrix, val_primitive_matrix, [], [], val_ground, []

        elif dataset == 'imdb':
            train_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_train.npy')
            train_primitive_matrix = np.array(train_primitive_matrix.item().todense())

            val_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_val.npy')
            val_primitive_matrix = np.array(val_primitive_matrix.item().todense())

            test_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_test.npy')
            test_primitive_matrix = np.array(test_primitive_matrix.item().todense())

            val_ground = np.load(data_path+dataset+'/ground_val.npy')
            train_ground = np.load(data_path+dataset+'/ground_train.npy')
            test_ground = np.load(data_path+dataset+'/ground_test.npy')

            return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground, val_ground, test_ground
        
        elif dataset == 'mscoco':
            train_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_train.npy')
            train_primitive_matrix = np.array(train_primitive_matrix.item().todense())

            val_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_val.npy')
            val_primitive_matrix = np.array(val_primitive_matrix.item().todense())

            test_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_test.npy')
            test_primitive_matrix = np.array(test_primitive_matrix.item().todense())

            val_ground = np.load(data_path+dataset+'/ground_val.npy')
            train_ground = np.load(data_path+dataset+'/ground_train.npy')
            test_ground = np.load(data_path+dataset+'/ground_test.npy')

            return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground, val_ground, test_ground
       

       



        
    