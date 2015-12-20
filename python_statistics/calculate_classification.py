from .calculate_dependencies import *
from .calculate_base import calculate_base
class calculate_classification(calculate_base):

    def calculate_svm(self):
        '''SVM
        http://scikit-learn.org/stable/modules/svm.html
        '''

    def calculate_randomForest(self):
        '''SVM
        http://scikit-learn.org/stable/modules/svm.html
        '''

    def calculate_decisionTree(self):
        '''
        sklearn.tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, presort=False)
        '''