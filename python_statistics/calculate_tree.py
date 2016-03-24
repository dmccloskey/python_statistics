from .calculate_dependencies import *
from .calculate_base import calculate_base

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
class calculate_tree(calculate_base):
    
    def make_dataModel_RandomForestClassifier(self):
        '''Random Forest
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
        sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        from sklearn.ensemble import RandomForestClassifier
        #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create Random Forest object
        model= RandomForestClassifier()
        # Train the model using the training sets and check score
        model.fit(X, y)
        #Predict Output
        predicted= model.predict(x_test)
        '''

    def make_dataModel_DecisionTreeClassifier(self):
        '''Decision Tree
        sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False)
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        INPUT:
        OUTPUT:
        EXAMPLE:
        '''

    def make_dataModel_ExtraTreesClassifier(self):
        '''ExtraTreesClassifier
        ExtraTreesClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
        INPUT:
        OUTPUT:
        EXAMPLE:
        '''

    def make_dataModel_DecisionTreeRegressor(self):
        '''Decision Tree
        sklearn.tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, presort=False)
        
        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        #Import other necessary libraries like pandas, np...
        from sklearn import tree
        #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create tree object 
        model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
        # model = tree.DecisionTreeRegressor() for regression
        # Train the model using the training sets and check score
        model.fit(X, y)
        model.score(X, y)
        #Predict Output
        predicted= model.predict(x_test)
        '''
    def make_dataModel_AdaBoostClassifier(self,
            data_I,):
        '''AdaBoostClassifier
        base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None

        EXAMPLE:
        http://scikit-learn.org/stable/modules/ensemble.html#adaboost

        http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#example-ensemble-plot-adaboost-multiclass-py
        bdt_real = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=2),
            n_estimators=600,
            learning_rate=1)
        '''
        pass;