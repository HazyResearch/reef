import numpy as np
import itertools

from sklearn.linear_model import LogisticRegression

class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """
    def __init__(self, primitive_matrix, val_ground,b=0.5, beta=0.2):
        """ 
        Initialize Synthesizer object

        b: class prior of most likely class (TODO: use somewhere)
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.b=b
        self.beta = beta

    def generate_feature_combinations(self, cardinality=1):
        """ 
        Create a list of primitive index combinations for given cardinality

        cardinality: number of features each heuristic operates over
        """
        primitive_idx = range(self.p)
        feature_combinations = []

        for comb in itertools.combinations(primitive_idx, cardinality):
            feature_combinations.append(comb)

        return feature_combinations

    def fit_function(self, comb):
        """ 
        Fits a single logistic regression model

        comb: feature combination to fit model over
        """
        X = self.val_primitive_matrix[:,comb]
        if np.shape(X)[0] == 1:
            X = X.reshape(-1,1)

        lr = LogisticRegression()
        lr.fit(X,self.val_ground)
        return lr

    def generate_heuristics(self, cardinality=1):
        """ 
        Generates heuristics over given feature cardinality

        cardinality: number of features each heuristic operates over
        """
        feature_combinations = self.generate_feature_combinations(cardinality)
        m = len(feature_combinations)

        heuristics = []
        for i,comb in enumerate(feature_combinations):
            heuristics.append(self.fit_function(comb))

        return heuristics, feature_combinations

    def beta_optimizer(self,marginals,abstain_weight=0.75):
        beta_params = np.linspace(0.0,0.45,10)
        accuracies_weighted = []

        for beta in beta_params:
            labels_cutoff = np.zeros(np.shape(marginals))
            labels_cutoff[marginals <= (self.b-beta)] = -1.
            labels_cutoff[marginals >= (self.b+beta)] = 1.

            coverage = np.mean(np.abs(labels_cutoff) != 0)
            accuracy = np.mean(labels_cutoff == self.val_ground)/coverage

            #import pdb; pdb.set_trace()
            accuracies_weighted.append(coverage*accuracy + (1-coverage)*abstain_weight)
        
        #import pdb; pdb.set_trace()
        return beta_params[np.argmax(np.array(accuracies_weighted))]


    def apply_heuristics(self, heuristics, X):
        """ 
        Generates heuristics over given feature cardinality

        heuristics: list of pre-trained logistic regression models
        X: primitive matrix to apply heuristics to
        """

        L = np.zeros((np.shape(X)[0],len(heuristics)))
        for i,hf in enumerate(heuristics):
            marginals = hf.predict_proba(X[:,i])[:,1]
            labels_cutoff = np.zeros(np.shape(marginals))
            #Goal is to maximize (C*A) + (1-C)*abstain_weight
            #TODO: Fix the beta optimizer - might only want accuracy or coverage?
            beta_temp = self.beta_optimizer(marginals, abstain_weight=0.0)
            labels_cutoff[marginals <= (self.b-beta_temp)] = -1.
            labels_cutoff[marginals >= (self.b+beta_temp)] = 1.
            L[:,i] = labels_cutoff
        return L

    def prune_heuristics(self,heuristics,feat_combos,keep=1):
        """ 
        Selects the best heuristic based on "bryan's metric"

        keep: number of heuristics to keep from all generated heuristics
        """
        L = self.apply_heuristics(heuristics,self.val_primitive_matrix[:,feat_combos])
        coverages = np.mean(np.abs(L.T) != 0, axis = 1)
        accuracies = np.mean(L.T == self.val_ground, axis=1)/coverages
        
        bm = [(a*b) + (0.5*(1-b)) for a,b in zip(accuracies,coverages)] 
        bm = np.nan_to_num(bm)

        
        sort_idx = np.argsort(bm)[::-1][0:keep]
        return sort_idx

#TODO: function for getting accuracies and TP FP rates





