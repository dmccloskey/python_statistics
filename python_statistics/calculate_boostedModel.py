from .calculate_dependencies import *
from .calculate_base import calculate_base

from sklearn.ensemble import GradientBoostingClassifier
#import xgboost as xgb

class calculate_boostedModel(calculate_base):

    def make_dataModel_GradientBoostingClassifier(self,):
        '''
        EXAMPLE:
        http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
        loss = 'deviance' logistic regression base estimator
        loss = 'exponential' equivalent of AdaBoost
        '''
        pass;
    def make_dataModel_xgboost(self,):
        '''
        EXAMPLE:
        https://www.kaggle.com/cast42/santander-customer-satisfaction/xgboost-with-early-stopping
        EXAMPLE:
        https://www.kaggle.com/srodriguex/santander-customer-satisfaction/model-and-feature-selection-with-python/notebook
        xgb.XGBClassifier()
        '''
        pass;
