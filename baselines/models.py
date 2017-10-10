import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

#TODO: make this clean!!
class BoostClassifier(object):
    """
    A class to go through the synthesizer-verifier loop
    """

    def __init__(self, train_primitive_matrix, val_primitive_matrix, 
    val_ground, train_ground=None, b=0.5):
        """ 
        Initialize DecisionTree object
        """

        self.train_primitive_matrix = train_primitive_matrix
        self.val_primitive_matrix = val_primitive_matrix
        self.val_ground = val_ground
        self.train_ground = train_ground
        self.b = b

    def fit(self):
        self.ab = AdaBoostClassifier(random_state=0)
        self.ab.fit(self.val_primitive_matrix, self.val_ground)


    def evaluate(self):
        """ 
        Calculate the accuracy and coverage for train and validation sets
        """
        self.val_marginals = self.ab.predict_proba(self.val_primitive_matrix)[:,1]
        self.train_marginals = self.ab.predict_proba(self.train_primitive_matrix)[:,1]

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

    import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(object):
    """
    A class to go through the synthesizer-verifier loop
    """

    def __init__(self, train_primitive_matrix, val_primitive_matrix, 
    val_ground, train_ground=None, b=0.5, depth=None):
        """ 
        Initialize DecisionTree object
        """

        self.train_primitive_matrix = train_primitive_matrix
        self.val_primitive_matrix = val_primitive_matrix
        self.val_ground = val_ground
        self.train_ground = train_ground
        self.b = b
        self.depth = np.shape(self.val_primitive_matrix)[1]

    def fit(self):
        self.dt = DecisionTreeClassifier(random_state=0, max_depth=self.depth)
        self.dt.fit(self.val_primitive_matrix, self.val_ground)


    def evaluate(self):
        """ 
        Calculate the accuracy and coverage for train and validation sets
        """
        self.val_marginals = self.dt.predict_proba(self.val_primitive_matrix)[:,1]
        self.train_marginals = self.dt.predict_proba(self.train_primitive_matrix)[:,1]

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