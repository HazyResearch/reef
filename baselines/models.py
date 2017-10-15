import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.linear_model import SGDClassifier

#need to clone https://github.com/tmadl/semisup-learn
#TODO: work in progress, waiting on Cmake
# import os
import sys
sys.path.append('/dfs/scratch0/paroma/semisup-learn/')
sys.path.append('/dfs/scratch0/paroma/nlopt/install/lib/python2.7/site-packages/')
# sys.path.append('/dfs/scratch0/paroma/include/')
from frameworks.CPLELearning import CPLELearningModel


class BaselineModel(object):
    """
    A base class for all sklearn-esque baseline methods
    """

    def __init__(self, train_primitive_matrix, val_primitive_matrix, 
    val_ground, train_ground=None, b=0.5):
        """ 
        Initialize DecisionTree object
        """

        self.model = None
        self.train_primitive_matrix = train_primitive_matrix
        self.val_primitive_matrix = val_primitive_matrix
        self.val_ground = val_ground
        self.train_ground = train_ground
        self.b = b

    def fit(self, model):
        pass

    def evaluate(self):
        """ 
        Calculate the accuracy and coverage for train and validation sets
        """
        self.val_marginals = self.model.predict_proba(self.val_primitive_matrix)[:,1]
        self.train_marginals = self.model.predict_proba(self.train_primitive_matrix)[:,1]

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


class BoostClassifier(BaselineModel):
    """
    AdaBoost Implementation
    """

    def fit(self):
        self.model = AdaBoostClassifier(random_state=0)
        self.model.fit(self.val_primitive_matrix, self.val_ground)

class DecisionTree(BaselineModel):
    """
    DecisionTree Implementation
    """

    def fit(self):
        self.model = DecisionTreeClassifier(random_state=0)
        self.model.fit(self.val_primitive_matrix, self.val_ground)

class SemiSupervised(BaselineModel):
    """
    LabelSpreading Implementation
    """

    def fit(self):
        #Need to concatenate labeled and unlabeled data
        #unlabeled data labels are set to -1
        X = np.concatenate((self.val_primitive_matrix, self.train_primitive_matrix))
        val_labels = (self.val_ground+1)/2.
        train_labels = -1.*np.ones(np.shape(self.train_ground))
        y = np.concatenate((val_labels, train_labels))

        self.model = LabelSpreading(kernel='knn')
        self.model.fit(X, y)

class ContrastiveSemiSupervised(BaselineModel):
    """
    Contrastive Pessimistic Likelihood Estimation Implementation
    """

    def fit(self):
        #First fit a LR model on the labeled data
        basemodel = SGDClassifier(loss='log', penalty='l1') # scikit logistic regression
        basemodel.fit(self.val_primitive_matrix,(self.val_ground+1)/2.)

        #Need to concatenate labeled and unlabeled data
        #unlabeled data labels are set to -1
        X = np.concatenate((self.val_primitive_matrix, self.train_primitive_matrix))
        val_labels = (self.val_ground+1)/2.
        train_labels = -1.*np.ones(np.shape(self.train_ground))
        y = np.concatenate((val_labels, train_labels))

        self.model = CPLELearningModel(basemodel)
        self.model.fit(X, y)


