import numpy as np
from sklearn.metrics import f1_score

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

        #TODO: remove taking beta in as a parameter or add a proper flag...
        self.beta = beta
        self.gamma = gamma
        self.vf = None
        self.syn = None

        self.hf = []
        self.feat_combos = []

    def prune_heuristics(self,heuristics,feat_combos,keep=1):
        """ 
        Selects the best heuristic based on Jaccard Distance and Reliability Metric

        keep: number of heuristics to keep from all generated heuristics
        """

        def calculate_jaccard_distance(num_labeled_total, num_labeled_L):
            scores = np.zeros(np.shape(num_labeled_L)[1])
            for i in range(np.shape(num_labeled_L)[1]):
                scores[i] = np.sum(np.minimum(num_labeled_L[:,i],num_labeled_total))/np.sum(np.maximum(num_labeled_L[:,i],num_labeled_total))
            return 1-scores

        #Note that the LFs are being applied to the entire val set though they were developed on a subset...
        L = self.syn.apply_heuristics(heuristics,self.val_primitive_matrix[:,feat_combos])

        #Use F1 trade-off for reliability
        acc_cov_scores = [f1_score(L[:,i], self.val_ground, average='micro') for i in range(np.shape(L)[1])] 
        acc_cov_scores = np.nan_to_num(acc_cov_scores)
        
        if self.vf != None:
            #Calculate Jaccard score for diversity
            val_num_labeled = np.sum(np.abs(self.vf.L_val.T), axis=0) 
            jaccard_scores = calculate_jaccard_distance(val_num_labeled,np.abs(L))
        else:
            jaccard_scores = np.ones(np.shape(acc_cov_scores))

        #Weighting the two scores to find best heuristic
        combined_scores = 0.5*acc_cov_scores + 0.5*jaccard_scores
        sort_idx = np.argsort(combined_scores)[::-1][0:keep]
        return sort_idx
     

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

        self.syn = Synthesizer(primitive_matrix, ground, beta=self.beta, b=self.b)
        hf, feat_combos = self.syn.generate_heuristics(cardinality)
        sort_idx = self.prune_heuristics(hf,feat_combos, keep)

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
            labels = np.sign(2*(marginals - 0.5))
            return np.sum(labels == ground)/float(total)
    
        def calculate_coverage(marginals, b, ground):
            #TODO: HOW DO I USE b!
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2*(marginals - 0.5))
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

    def gamma_optimizer_old(self,marginals,abstain_weight=0.8):
        """ 
        Returns the best gamma parameter for abstain threshold given marginals

        marginals: confidences for data from a single heuristic
        abstain_weight: weight to give abstains for Bryan's Metric
        """
        gamma_params = np.linspace(0.0,0.45,10)
        accuracies_weighted = []

        for gamma in gamma_params:
            labels_cutoff = np.zeros(np.shape(marginals))
            labels_cutoff[marginals <= (self.b-gamma)] = -1.
            labels_cutoff[marginals >= (self.b+gamma)] = 1.

            coverage = np.mean(np.abs(labels_cutoff) != 0)
            accuracy = np.mean(labels_cutoff == self.val_ground)/coverage
            accuracies_weighted.append(coverage*accuracy + (1-coverage)*abstain_weight)
        
        #import pdb; pdb.set_trace()
        accuracies_weighted = np.nan_to_num(accuracies_weighted)
        return gamma_params[np.argmax(np.array(accuracies_weighted))]

    def gamma_optimizer(self,marginals,abstain_weight=0.8):
        """ 
        Returns the best gamma parameter for abstain threshold given marginals

        marginals: confidences for data from a single heuristic
        abstain_weight: weight to give abstains for Bryan's Metric
        """
        m = len(self.hf)
        gamma = 0.5-(1/(m**(3/2.)))
        return gamma

    def find_feedback(self):
        """ 
        Finds vague points according to gamma parameter

        self.gamma: confidence past 0.5 that relates to a vague or incorrect point
        """
        #TODO: flag for re-classifying incorrect points
        #incorrect_idx = self.vf.find_incorrect_points(b=self.b)

        gamma_opt = self.gamma_optimizer(self.vf.val_marginals)
        #gamma_opt = self.gamma
        vague_idx = self.vf.find_vague_points(b=self.b, gamma=gamma_opt)
        incorrect_idx = vague_idx
        self.feedback_idx = list(set(list(np.concatenate((vague_idx,incorrect_idx)))))    