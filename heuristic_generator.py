import numpy as np

from program_synthesis.synthesizer import Synthesizer
from program_synthesis.verifier import Verifier

class HeuristicGenerator(object):
    """
    A class to go through the synthesizer-verifier loop
    """

    def __init__(self, train_primitive_matrix, val_primitive_matrix, 
    val_ground, train_ground=None, b=0.5, beta=0.2, gamma=0.35):
        """ 
        Initialize HeuristicGenerator object

        b: class prior of most likely class (TODO: use somewhere)
        beta: threshold to decide whether to abstain or label for heuristics
        gamma: threshold to decide whether to call a point vague or not
        """

        self.train_primitive_matrix = train_primitive_matrix
        self.val_primitive_matrix = val_primitive_matrix
        self.val_ground = val_ground
        self.train_ground = train_ground
        self.b = b
        self.beta = beta
        self.gamma = gamma

        self.hf = []
        self.feat_combos = []

    def run_synthesizer(self, cardinality=1, idx=None, keep=1):
        """ 
        Generates Synthesizer object and saves all generated heuristics

        cardinality: number of features candidate programs take as input
        idx: indices of validation set to fit programs over
        keep: number of heuristics to pass to verifier
        """
        if idx == None:
            primitive_matrix = self.val_primitive_matrix
            ground = self.val_ground
        else:
            primitive_matrix = self.val_primitive_matrix[idx,:]
            ground = self.val_ground[idx]

        self.syn = Synthesizer(primitive_matrix, ground,b=self.b)
        hf, feat_combos = self.syn.generate_heuristics(cardinality)
        sort_idx = self.syn.prune_heuristics(hf,feat_combos, keep)

        for i in sort_idx:
            self.hf.append(hf[i]) 
            self.feat_combos.append(feat_combos[i])

        self.X_train = self.train_primitive_matrix[:,self.feat_combos]
        self.L_train = self.syn.apply_heuristics(self.hf,self.X_train)
        self.X_val = self.val_primitive_matrix[:,self.feat_combos]
        self.L_val = self.syn.apply_heuristics(self.hf,self.X_val)

    def evaluate(self):
        """ 
        Calculate the accuracy and coverage for train and validation sets
        """
        self.val_marginals = self.vf.val_marginals
        self.train_marginals = self.vf.train_marginals

        def calculate_accuracy(marginals, b, ground):
            #TODO: HOW DO I USE b!
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.round(2*(marginals - 0.5))
            return np.sum(labels == ground)/float(total)
    
        def calculate_coverage(marginals, b, ground):
            #TODO: HOW DO I USE b!
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.round(2*(marginals - 0.5))
            return total/float(len(labels))

        
        self.val_accuracy = calculate_accuracy(self.val_marginals, self.b, self.val_ground)
        self.train_accuracy = calculate_accuracy(self.train_marginals, self.b, self.train_ground)
        self.val_coverage = calculate_coverage(self.val_marginals, self.b, self.val_ground)
        self.train_coverage = calculate_coverage(self.train_marginals, self.b, self.train_ground)
        return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage
    
    def run_verifier(self):
        """ 
        Generates Verifier object and saves marginals
        """
        self.vf = Verifier(self.L_train, self.L_val, self.val_ground)
        self.vf.train_gen_model()
        self.vf.assign_marginals()

    def find_feedback(self):
        """ 
        Finds vague points according to gamma parameter

        self.gamma: confidence past 0.5 that relates to a vague or incorrect point
        """
        vague_idx = self.vf.find_vague_points(b=self.b, thresh=self.gamma)
        #TODO: flag for re-classifying incorrect points
        #incorrect_idx = self.vf.find_incorrect_points(b=self.b)
        incorrect_idx = vague_idx
        self.feedback_idx = list(set(list(np.concatenate((vague_idx,incorrect_idx)))))






















    