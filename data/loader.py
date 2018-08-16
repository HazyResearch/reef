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
        if dataset == 'bone_tumor':
            primitive_matrix = np.load(data_path+dataset+'/primitive_matrix.npy')
            features = np.load(data_path+dataset+'/features.npy')
            ground = np.load(data_path+dataset+'/ground.npy')

            train_primitive_matrix = primitive_matrix[0:400,:]
            train_feature_matrix = features[0:400,:]
            train_ground = ground[0:400]

            val_primitive_matrix = primitive_matrix[400:600,:]
            val_feature_matrix = features[400:600,:]
            val_ground = ground[400:600]

            test_primitive_matrix = primitive_matrix[600:802,:]
            test_feature_matrix = features[600:802,:]
            test_ground = ground[600:802]

            return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground, val_ground, test_ground
       
        elif dataset == 'mammogram':
            primitive_matrix = np.load(data_path+dataset+'/primitive_matrix.npy')
            primitive_matrix = primitive_matrix.T
            ground = np.load(data_path+dataset+'/ground.npy')

            train_primitive_matrix = primitive_matrix[0:1488,:]
            train_ground = ground[0:1488]
            val_primitive_matrix = primitive_matrix[1488:1674,:]
            val_ground = ground[1488:1674]
            return primitive_matrix, train_primitive_matrix, val_primitive_matrix, [], train_ground, val_ground, []
    
            # train_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_train.npy')
            # val_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_val.npy')
            # test_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_test.npy')

            # train_ground = np.load(data_path+dataset+'/ground_train.npy')
            # val_ground = np.load(data_path+dataset+'/ground_val.npy')
            # test_ground = np.load(data_path+dataset+'/ground_test.npy')

            # return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground, val_ground, test_ground
        
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

            common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
            return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], [], [], val_ground, []

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

            common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
            return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix, train_ground, val_ground, test_ground
        
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

            common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix, thresh=0.05)

            return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix, train_ground, val_ground, test_ground

        elif dataset == 'mobile_mb':
            train_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_train.npy')
            val_primitive_matrix = np.load(data_path+dataset+'/primitive_matrix_val.npy')

            train_ground = np.load(data_path+dataset+'/ground_train.npy')
            val_ground = np.load(data_path+dataset+'/ground_val.npy')

            return train_primitive_matrix, val_primitive_matrix, [], train_ground, val_ground, []

        elif dataset == 'discovery':
            primitive_matrix = np.load(data_path+dataset+'/primitive_matrix.npy')
            ground = np.load(data_path+dataset+'/ground.npy')

            #Some data processing in here
            positive_indices = np.where(ground == 1.)[0]
            negative_indices = np.where(ground == 0.)[0]
            ground[negative_indices] = -1.

            #To test scaling and class balance issues
            num_positive = 50
            num_negative = 50
            train_val_mult = 5

            #Randomly sample indices
            positive_choose = np.random.choice(positive_indices, (train_val_mult+1)*num_positive, replace=False)
            negative_choose = np.random.choice(negative_indices, (train_val_mult+1)*num_negative, replace=False)


            #Create train and validation sets (NO TEST FOR NOW)
            val_primitive_matrix = np.concatenate((primitive_matrix[positive_choose[0:num_positive],:], primitive_matrix[negative_choose[0:num_negative],:]))
            val_ground = np.concatenate((ground[positive_choose[0:num_positive]], ground[negative_choose[0:num_negative]]))

            train_primitive_matrix = np.concatenate((primitive_matrix[positive_choose[num_positive:(train_val_mult+1)*num_positive],:], primitive_matrix[negative_choose[num_negative:(train_val_mult+1)*num_negative],:]))
            train_ground = np.concatenate((ground[positive_choose[num_positive:(train_val_mult+1)*num_positive]], ground[negative_choose[num_negative:(train_val_mult+1)*num_negative]]))

            return train_primitive_matrix, val_primitive_matrix, [], train_ground, val_ground, []

        elif dataset == 'cdr':
            train_ground = np.load(data_path+dataset+'/ground_train.npy')
            train_ground[np.where(train_ground == 0.)] = -1.
            val_ground = np.load(data_path+dataset+'/ground_val.npy')
            val_ground[np.where(val_ground == 0.)] = -1.

            train_features = np.load(data_path+dataset+'/split0_features.npz')
            train_primitive_matrix =  np.array(scipy.sparse.csr_matrix((train_features['data'], train_features['indices'], train_features['indptr'])).todense()).astype(float)

            val_features = np.load(data_path+dataset+'/split1_features.npz')
            val_primitive_matrix =  np.array(scipy.sparse.csr_matrix((val_features['data'], val_features['indices'], val_features['indptr'])).todense()).astype(float)

            common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix, thresh=0.01)

            return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], [], train_ground, val_ground, []




        
    