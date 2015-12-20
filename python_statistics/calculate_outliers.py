from .calculate_dependencies import *
from .calculate_base import calculate_base
import itertools
from listDict.listDict import listDict
import copy

def _rsd(data_I):
    '''calculate the relative standard deviation (RSD)
    INPUT:
    data_I = numpy array
    '''
    mean = numpy.mean(data_I);
    std = numpy.std(data_I);
    rsd = std/mean;
    return rsd;

class calculate_outliers(calculate_base):
    def calculate_outliers(self,data_I,key_I,deviation_I,method_I='cv',data_labels_I = 'sample_name_short'):
        '''calculate outliers in a 1D data set
        METHOD:
        compare the relative deviation change after removing a subset of points
        to a user defined deviation threshold.

        INPUT:
        data_I = listDict
        key_I = dictionary key
        deviation_I = float, threshold of what constitutes an outlier
        method_I = string, function to calculate the deviation in the data
            "cv" or "rsd" = coefficient of variation or relative standard deviation (mean/stdev)
            "stdev" = standard deviation
            "var" = variance
            default = "cv"
        data_labels_I = dictionary key for the data labels

        OUTPUT:
        outliers_O = listDict with additional fields
            outlier = boolean
            relative_deviation_change = float
            subset = tuple

        '''

        # extract out the data
        listdict = listDict(data_I);
        data_1 = listdict.extract_arrayFromListDict(key_I);

        # generate the deviation function
        if method_I == 'cv' or method_I == 'rsd':
            method = _rsd;
        elif method_I == 'stdev':
            method = numpy.std;
        elif method_I == 'var':
            method = numpy.std;
        else:
            print('Method not recognized.');
            print('default to "cv".');
            method = _rsd;

        outliers_O = [];

        # determine the initial deviation of the data
        data_1_dev = method(data_1);
        # determine the influence of each data point
        # or subset of data points on the variance of the data
        ndata_1 = len(data_1);
        data_1_index = [i for i in range(ndata_1)];
        maxpoints = ndata_1//2;
        # iterate over each number of points
        for np in range(maxpoints):
            # generate all combinations for each number of points
            for subset in itertools.combinations(data_1_index,np):
                test = numpy.delete(data_1,subset,axis=0);
                # calculate the change in variance without the subset
                test_dev = method(test);
                change = numpy.abs(test_dev-data_1_dev)/data_1_dev;
                # test and record outliers
                if change > deviation_I:
                    subset_names = [];
                    for i in subset:
                        subset_names.append(data_I[i][data_labels_I]);
                    for i in subset:
                        d = copy.copy(data_I[i]);
                        d['outlier']=True;
                        d['relative_deviation_change']=change;
                        d['subset_index']=subset;
                        d['subset_names']=subset_names;
                        d['outlier_method']=method_I;
                        d['outlier_deviation']=deviation_I;
                        outliers_O.append(d);

        return outliers_O;



