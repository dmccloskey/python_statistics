from .calculate_dependencies import *
from .calculate_base import calculate_base

from sklearn.cross_validation import train_test_split

class data_partitioning(calculate_base):
    def make_trainTestSplit(self,
            data_X_I,data_y_I=None,data_z_I=None,
            test_size_I=None,train_size_I=None,random_state_I=None):
        '''split the data into training and testing
        INPUT:
        data_X_I = numpy.array of dim: [nsamples,nfeatures]
        data_y_I = numpy array of dim: [nsamples,] (classes/factors/responses)
        data_z_I = numpy array of dim: [nsamples,] (indexes)
        INPUT train_test_split:
        ...

        '''

        #break into training and testing data splits
        if not data_y_I is None and not data_z_I is None:
            X,y,z = data_X_I,data_y_I,data_z_I;
            X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z,
                                                    test_size = test_size_I,
                                                    train_size = train_size_I,
                                                    random_state=random_state_I
                                                    );
            self.data_train['data'],self.data_test['data']=X_train, X_test;
            self.data_train['response'],self.data_test['response']=y_train, y_test;
            self.data_train['row_indexes'],self.data_test['row_indexes']=z_train, z_test;
            self.map_trainTestResponse2RowLabels();
        elif not data_y_I is None:
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

    def map_trainTestResponse2RowLabels(self):
        '''
        map the shuffled responses to the data row labels
        OUTPUT:
        row_labels_O = list
        '''
        
        self.data_train['row_labels'] = self.data['row_labels'].iloc[self.data_train['row_indexes']]
        self.data_test['row_labels'] = self.data['row_labels'].iloc[self.data_test['row_indexes']]