from .calculate_dependencies import *
from .calculate_base import calculate_base
class data_preProcessing(calculate_base):
    def impute_missingValues(self,):
        '''
        EXAMPLE:
        http://scikit-learn.org/stable/auto_examples/missing_values.html#example-missing-values-py
        '''
        pass;
    def centerAndScale_data(self,):
        '''
        Mean center and UV scale the data
        '''
        data_mean = self.data_train['data'].mean(axis=0);
        data_std = self.data_train['data'].std(axis=0);
        self.data_train['data'] = (self.data_train['data'] - data_mean) / data_std;