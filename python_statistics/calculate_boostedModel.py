from .calculate_dependencies import *
from .calculate_base import calculate_base

from sklearn.ensemble import GradientBoostingClassifier
#import xgboost as xgb

class calculate_boostedModel(calculate_base):

    def calculate_GradientBoostingClassifier(self,):
        '''
        EXAMPLE: maybe some issues with this code...
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        #Import Library
        from sklearn.ensemble import GradientBoostingClassifier
        #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create Gradient Boosting Classifier object
        model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        # Train the model using the training sets and check score
        model.fit(X, y)
        #Predict Output
        predicted= model.predict(x_test)
        '''
        pass;
    def calculate_AdaBoostClassifier(self,data_I,base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None):
        '''
        EXAMPLE:
        http://scikit-learn.org/stable/modules/ensemble.html#adaboost

        http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#example-ensemble-plot-adaboost-multiclass-py
        bdt_real = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=2),
            n_estimators=600,
            learning_rate=1)
        '''
        pass;
    def calculate_xgboost(self,):
        '''
        EXAMPLE:
        https://www.kaggle.com/cast42/santander-customer-satisfaction/xgboost-with-early-stopping
        EXAMPLE:
        https://www.kaggle.com/srodriguex/santander-customer-satisfaction/model-and-feature-selection-with-python/notebook
        xgb.XGBClassifier()
        '''
        pass;
