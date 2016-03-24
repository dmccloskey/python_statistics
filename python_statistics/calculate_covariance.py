from .calculate_dependencies import *
from .calculate_dataSplit import calculate_base
from sklearn.covariance import EmpiricalCovariance, MinCovDet
class calculate_covariance(calculate_base):
    def make_dataModel_MinCovDet(self,
        params_I = {'store_precision':True, 'assume_centered':False, 'support_fraction':None}):
        '''
        Minimum Covariance Determinant (MCD): robust estimator of covariance.
        sklearn.covariance.MinCovDet(store_precision=True, assume_centered=False, support_fraction=None, random_state=None)
        http://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html#sklearn.covariance.MinCovDet
        '''
        # fit a Minimum Covariance Determinant (MCD) robust estimator to data
        robust_cov = MinCovDet(random_state = self.random_state, **params_I)
        robust_cov.fit(self.data_train['data']);

        # store the model
        self.data_model = robust_cov;

    def make_dataModel_EmpiricalCovariance(self,
        params_I={'store_precision':True, 'assume_centered':False}):
        '''
        Maximum likelihood covariance estimator
        sklearn.covariance.EmpiricalCovariance(store_precision=True, assume_centered=False)
        http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance
        '''
        # compare estimators learned from the full data set with true parameters
        emp_cov = EmpiricalCovariance(**params_I)
        emp_cov.fit(self.data_train['data']);

        # store the model
        self.data_model = emp_cov;

    def extract_covarianceInformation(self,data_model_I=None):
        '''
        INPUT:
        data_model_I = covariance data model
        '''
        if data_model_I: data_model=data_model_I;
        else: data_model = self.data_model;

        mahal_data_model = data_model.mahalanobis(self.data_train['data']).ravel(); #dim: [nsamples]
        loglik_data_model = data_model.score(self.data_train['data']); #log-likelihood estimation
        pvalue_data_model = np.exp(loglik_data_model);
        cov_matrix = data_model.covariance_;
        precision_matrix = data_model.precision_;

        # return results
        return cov_matrix, precision_matrix, mahal_data_model, loglik_data_model, pvalue_data_model;
