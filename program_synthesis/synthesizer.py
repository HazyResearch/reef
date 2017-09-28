import numpy as np
import itertools

from sklearn.linear_model import LogisticRegression

class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """
    def __init__(self, primitive_matrix, val_ground,b=0.5, cutoff=0.2):
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.b=b
        self.cutoff = cutoff

    def generate_feature_combinations(self, cardinality=1):
        """ Create a list of primitive index combinations for given cardinality

        cardinality: number of features each heuristic operates over
        """
        primitive_idx = range(self.p)
        feature_combinations = []

        for comb in itertools.combinations(primitive_idx, cardinality):
            feature_combinations.append(comb)

        return feature_combinations

    def fit_function(self, comb):
        """ Fits a single logistic regression model

        comb: feature combination to fit model over
        """
        X = self.val_primitive_matrix[:,comb]
        if np.shape(X)[0] == 1:
            X = X.reshape(-1,1)

        lr = LogisticRegression()
        lr.fit(X,self.val_ground)
        return lr

    def generate_heuristics(self, cardinality=1):
        """ Generates heuristics over given feature cardinality

        cardinality: number of features each heuristic operates over
        """
        feature_combinations = self.generate_feature_combinations(cardinality)
        m = len(feature_combinations)

        heuristics = []
        for i,comb in enumerate(feature_combinations):
            heuristics.append(self.fit_function(comb))

        return heuristics, feature_combinations

    def apply_heuristics(self, heuristics, X):
        """ Generates heuristics over given feature cardinality

        heuristics: list of pre-trained logistic regression models
        X: primitive matrix to apply heuristics to
        """
        #TODO: check that X and heuristic shapes match!
        L = np.zeros((np.shape(X)[0],len(heuristics)))
        for i,hf in enumerate(heuristics):
            marginals = hf.predict_proba(X[:,i])[:,1]
            labels_cutoff = np.zeros(np.shape(marginals))
            labels_cutoff[marginals <= (self.b-self.cutoff)] = -1.
            labels_cutoff[marginals >= (self.b+self.cutoff)] = 1.
            L[:,i] = labels_cutoff
        return L

    def prune_heuristics(self,heuristics,feat_combos,keep=30):
        L = self.apply_heuristics(heuristics,self.val_primitive_matrix[:,feat_combos])
        accuracies = np.mean(L.T == self.val_ground, axis=1)
        sort_idx = np.argsort(accuracies)[::-1][0:keep]
        return sort_idx



#TODO: function for getting accuracies and TP FP rates





