from .calculate_dependencies import *
from .calculate_base import calculate_base

from sklearn.cross_validation import train_test_split

class calculate_dataSplit(calculate_base):
    def make_trainTestSplit(self,data_X_I,data_y_I=None,test_size_I=None,train_size_I=None,random_state_I=None):
        '''split the data into training and testing
        INPUT:
        data_X_I = numpy.array of dim: [nsamples,nfeatures]
        data_y_I = numpy array of dim: [nsamples,] (classes/factors/responses)
        INPUT train_test_split:
        ...

        '''

        #break into training and testing data splits
        if not data_y_I is None:
            X,y = data_X_I,data_y_I;
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = test_size_I,
                                                    train_size = train_size_I,
                                                    random_state=random_state_I
                                                    );
            self.data_train['data'],self.data_test['data']=X_train, X_test;
            self.data_train['response'],self.data_test['response']=y_train, y_test;
        else:
            X = data_X_I;
            X_train, X_test = train_test_split(X,
                                                    test_size = test_size_I,
                                                    train_size = train_size_I,
                                                    random_state=random_state_I
                                                    );
            self.data_train['data'],self.data_test['data']=X_train, X_test;