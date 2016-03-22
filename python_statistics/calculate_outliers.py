from .calculate_dependencies import *
from .calculate_dataSplit import calculate_base
import itertools
from listDict.listDict import listDict
import copy

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.covariance import EmpiricalCovariance, MinCovDet

def _rsd(data_I):
    '''calculate the relative standard deviation (RSD)
    INPUT:
    data_I = numpy array
    '''
    mean = np.mean(data_I);
    std = np.std(data_I);
    rsd = std/mean;
    return rsd;

class calculate_outliers(calculate_base):
    def calculate_outliers_deviation(self,data_I,key_I,deviation_I,method_I='cv',data_labels_I = 'sample_name_short'):
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
            method = np.std;
        elif method_I == 'var':
            method = np.std;
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
                test = np.delete(data_1,subset,axis=0);
                # calculate the change in variance without the subset
                test_dev = method(test);
                change = np.abs(test_dev-data_1_dev)/data_1_dev;
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

    def calculate_outliers_OneClassSVM(self,outlier_fraction_I=0.05,
            params_I = {'kernel':'rbf', 'degree':3, 'gamma':'auto', 'coef0':0.0, 'tol':0.001, 'nu':0.5, 'shrinking':True, 'cache_size':200, 'verbose':False, 'max_iter':-1, 'random_state':None}
            ):
        '''
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM
        http://scikit-learn.org/stable/modules/svm.html#svm-outlier-detection
        INPUT:
        outlier_fraction_I = fraction of expected outliers in the data
        DEFAULT INPUT:
        kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=None
        EXAMPLE:
        http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html#example-covariance-plot-outlier-detection-py
        '''
        clf = svm.OneClassSVM(**params_I);
        clf.fit(self.data_train['data']);
        # calculate the distance to the hyperplane
        distance = clf.decision_function(self.data_train['data']).ravel();
        outliers_indexes = calculate_outliers_decisionFunctionDistance(distance,outlier_fraction_I);
        return outliers_indexes;

    def calculate_outliers_EllipticEnvelope(self,outlier_fraction_I=0.05,
            params_I = {'store_precision':True, 'assume_centered':False, 'support_fraction':None, 'contamination':0.1, 'random_state':None}):
        '''
        http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope
        http://scikit-learn.org/stable/modules/outlier_detection.html#outlier-detection
        INPUT:
        DEFAULT INPUT:
        store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1, random_state=None
        EXAMPLE:
        http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html#example-covariance-plot-outlier-detection-py
        NOTES:
        Outlier detection from covariance estimation may break or not perform well in high-dimensional settings. 
        In particular, one will always take care to work with n_samples > n_features ** 2
        '''
        clf = EllipticEnvelope(**params_I);
        clf.fit(self.data_train['data']);
        # calculate the distance to the hyperplane
        distance = clf.decision_function(self.data_train['data']).ravel();
        outliers_indexes = calculate_outliers_decisionFunctionDistance(distance,outlier_fraction_I);
        return outliers_indexes;

    def calculate_outliers_decisionFunctionDistance(self,
            distance_I,outlier_fraction_I):
        '''
        http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope
        http://scikit-learn.org/stable/modules/outlier_detection.html#outlier-detection
        EXAMPLE:
        http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html#example-covariance-plot-outlier-detection-py
        NOTES:
        Outlier detection from covariance estimation may break or not perform well in high-dimensional settings. 
        In particular, one will always take care to work with n_samples > n_features ** 2
        '''
        # calculate the threshold for the outliers
        threshold = np.percentile(distance_I,100 * outlier_fraction_I);
        # get the outliers
        outliers_boolean = distance_I < threshold;
        outliers_data = self.data_train[outliers_boolean];
        outliers_indexes = [i for i,b in enumerate(outliers_boolean) if b];
        return outliers_indexes;

    def compare_mahalanobisDistances(self,
                                     ):
        '''
        Compare the mahalanobis distances
        calculated using a robust and non-robust outlier method
        '''
        # fit a Minimum Covariance Determinant (MCD) robust estimator to data
        robust_cov = MinCovDet().fit(self.data_train['data']);
        mahal_robust_cov = robust_cov.mahalanobis(self.data_train['data']).ravel(); #dim: [nsamples]
        loglik_robust_cov = robust_cov.score(self.data_train['data']);#log-likelihood estimation
        pvalue_robust_cov = np.exp(loglik_robust_cov);
        # compare estimators learned from the full data set with true parameters
        emp_cov = EmpiricalCovariance().fit(self.data_train['data']);
        mahal_emp_cov = emp_cov.mahalanobis(self.data_train['data']).ravel();
        loglik_emp_cov = emp_cov.score(self.data_train['data'])
        pvalue_emp_cov = np.exp(loglik_emp_cov);

        #plt.scatter(mahal_robust_cov,mahal_emp_cov);
        #plt.show()

        return mahal_robust_cov,mahal_emp_cov,pvalue_robust_cov,pvalue_emp_cov;
