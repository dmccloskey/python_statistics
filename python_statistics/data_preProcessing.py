from .calculate_dependencies import *
from .calculate_base import calculate_base
from sklearn.preprocessing import label_binarize
class data_preProcessing(calculate_base):
    def impute_missingValues(self,):
        '''
        EXAMPLE:
        http://scikit-learn.org/stable/auto_examples/missing_values.html#example-missing-values-py
        '''
        pass;
    def centerAndScale_data(self,):
        '''
        Mean center and UV scale the training and test data
        '''
        #training data
        data_mean = self.data_train['data'].mean(axis=0);
        data_std = self.data_train['data'].std(axis=0);
        self.data_train['data'] = (self.data_train['data'] - data_mean) / data_std;
        #testing data
        data_mean = self.data_test['data'].mean(axis=0);
        data_std = self.data_test['data'].std(axis=0);
        self.data_test['data'] = (self.data_test['data'] - data_mean) / data_std;

    def convert_factor2DummyResponse(self,):
        '''
        Convert factor vector to a dummy response matrix
        NOTES: needed to implement plsda
        EXAMPLES:
        >>> from sklearn.preprocessing import label_binarize
        >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
        array([[1, 0, 0, 0],
               [0, 0, 0, 1]])
        INPUT:
        OUTPUT:
        '''
        factors = self.get_uniqueResponses();
        dummyResponse = label_binarize(factors, classes=self.data_train['response']);
        self.data_train['response'] = dummyResponse;
