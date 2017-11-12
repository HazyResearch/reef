import numpy as np
import itertools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """
    def __init__(self, primitive_matrix, val_ground,b=0.5):
        """ 
        Initialize Synthesizer object

        b: class prior of most likely class
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.b=b

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

    def fit_function(self, comb, model):
        """ 
        Fits a single logistic regression or decision tree model

        comb: feature combination to fit model over
        model: fit logistic regression or a decision tree
        """
        X = self.val_primitive_matrix[:,comb]
        if np.shape(X)[0] == 1:
            X = X.reshape(-1,1)

        # fit decision tree or logistic regression
        if model == 'dt':
            dt = DecisionTreeClassifier(max_depth=len(comb))
            dt.fit(X,self.val_ground)
            return dt

        elif model == 'lr':
            lr = LogisticRegression()
            lr.fit(X,self.val_ground)
            return lr

    def generate_heuristics(self, model, cardinality=1):
        """ 
        Generates heuristics over given feature cardinality

        model: fit logistic regression or a decision tree
        cardinality: number of features each heuristic operates over
        """
        feature_combinations = self.generate_feature_combinations(cardinality)
        m = len(feature_combinations)

        heuristics = []
        for i,comb in enumerate(feature_combinations):
            heuristics.append(self.fit_function(comb, model))

        return heuristics, feature_combinations

    def beta_optimizer_old(self,marginals):
        """ 
        Returns the best beta parameter for abstain threshold given marginals
        Uses some random Paroma metric to decide beta, but it works well

        marginals: confidences for data from a single heuristic
        """

        #Set the range of beta params
        #0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = np.linspace(0.25,0.45,5)

        confidence = np.abs(marginals-self.b)
        confidence_mean = np.mean(confidence)
        confidence_max = np.max(confidence)
        return beta_params[np.argmin(np.abs(confidence_mean-beta_params))]

    def beta_optimizer(self,marginals, ground):
        """ 
        Returns the best beta parameter for abstain threshold given marginals
        Uses F1 score that maximizes the F1 score

        marginals: confidences for data from a single heuristic
        """

        #Set the range of beta params
        #0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = np.linspace(0.25,0.45,10)

        f1 = []		
 		
        for beta in beta_params:		
            labels_cutoff = np.zeros(np.shape(marginals))		
            labels_cutoff[marginals <= (self.b-beta)] = -1.		
            labels_cutoff[marginals >= (self.b+beta)] = 1.		
 		
            #coverage = np.mean(np.abs(labels_cutoff) != 0)		
            #accuracy = np.mean(labels_cutoff == self.val_ground)/coverage	
            #import pdb; pdb.set_trace()
            #assert(len(self.val_ground) == len(labels_cutoff))
            f1.append(f1_score(ground, labels_cutoff, average='micro'))
         		
        f1 = np.nan_to_num(f1)		
        return beta_params[np.argmax(np.array(f1))]


    def find_optimal_beta(self, heuristics, X, ground):
        """ 
        Returns optimal beta for given heuristics

        heuristics: list of pre-trained logistic regression models
        X: primitive matrix to apply heuristics to
        ground: ground truth associated with X data
        """

        beta_opt = []
        for i,hf in enumerate(heuristics):
            marginals = hf.predict_proba(X[:,i])[:,1]
            labels_cutoff = np.zeros(np.shape(marginals))
            beta_opt.append((self.beta_optimizer(marginals, ground)))
        return beta_opt

#TODO: function for getting accuracies and TP FP rates





