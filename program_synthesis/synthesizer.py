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

        b: class prior of most likely class
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

    def beta_optimizer(self,marginals):
        """ 
        Returns the best beta parameter for abstain threshold given marginals

        marginals: confidences for data from a single heuristic
        """

        #Set the range of beta params
        #0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = np.linspace(0.25,0.45,5)

        confidence = np.abs(marginals-self.b)
        confidence_mean = np.mean(confidence)
        confidence_max = np.max(confidence)
        return beta_params[np.argmin(np.abs(confidence_mean-beta_params))]


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
            beta_opt = self.beta_optimizer(marginals)

            labels_cutoff[marginals <= (self.b-beta_opt)] = -1.
            labels_cutoff[marginals >= (self.b+beta_opt)] = 1.
            L[:,i] = labels_cutoff
        return L

#TODO: function for getting accuracies and TP FP rates





