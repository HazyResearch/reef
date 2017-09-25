import numpy as np
from scipy import sparse

from snorkel.learning import GenerativeModel
from snorkel.learning import RandomSearch
from snorkel.learning.structure import DependencySelector

class Verifier(object):
    """
    A class for the Snorkel Model Verifier
    """

    def __init__(self, L_train, L_val, val_ground):
        self.L_train = L_train.astype(int)
        self.L_val = L_val.astype(int)
        self.val_ground = val_ground

    def train_gen_model(self,deps=False,grid_search=False):
        """ Trains generative model acc. to parameters

        deps,grid_search: flags for generative model
        """
        if not grid_search:
            #TODO: Include dependencies option
            gen_model = GenerativeModel()
            gen_model.train(self.L_train, epochs=100, decay=0.001 ** (1.0 / 100), step_size=0.005, reg_param=1.0)
        self.gen_model = gen_model

    def assign_marginals(self):
        """ Assigns probabilistic labels for train and val sets """ 
        self.train_marginals = self.gen_model.marginals(sparse.csr_matrix(self.L_train))
        self.val_marginals = self.gen_model.marginals(sparse.csr_matrix(self.L_val))

    def find_vague_points(self,thresh=0.1,b=0.5):
        """ Find val set indices where marginals are within thresh of b """
        val_idx = np.where(np.abs(self.val_marginals-b) <= thresh)
        return val_idx

    def find_incorrect_points(self,b=0.5):
        """ Find val set indices where marginals are incorrect """
        val_labels = 2*(self.val_marginals > b)-1
        val_idx = np.where(val_labels != self.val_ground)
        return val_idx




       

        

    


        
