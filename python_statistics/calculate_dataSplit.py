from .calculate_dependencies import *
from .calculate_base import calculate_base

from sklearn.cross_validation import train_test_split

class calculate_dataSplit(calculate_base):
    def calculate_trainTestSplit(self,data_I,random_state):
        '''split the data into training and testing
        '''

        #break into training and testing data splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

        #score the model

        #validate the model
        #cross validation the model
        #permute the model
        #make the final model
        pass;