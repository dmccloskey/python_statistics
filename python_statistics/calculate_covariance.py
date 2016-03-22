from .calculate_dependencies import *
from .calculate_dataSplit import calculate_base
from sklearn.covariance import EmpiricalCovariance, MinCovDet
class calculate_covariance(calculate_base):
    def calculate_MinCovDet(self,
        params_I = {'store_precision':True, 'assume_centered':False, 'support_fraction':None, 'random_state':self.random_state}):
        '''
        Minimum Covariance Determinant (MCD): robust estimator of covariance.
        sklearn.covariance.MinCovDet(store_precision=True, assume_centered=False, support_fraction=None, random_state=None)
        http://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html#sklearn.covariance.MinCovDet
        '''
        # fit a Minimum Covariance Determinant (MCD) robust estimator to data
        robust_cov = MinCovDet(**params_I)
        robust_cov.fit(self.data_train['data']);
        mahal_robust_cov = robust_cov.mahalanobis(self.data_train['data']).ravel(); #dim: [nsamples]
        loglik_robust_cov = robust_cov.score(self.data_train['data']);#log-likelihood estimation
        pvalue_robust_cov = np.exp(loglik_robust_cov);
        cov_matrix = robust_cov.covariance_;
        precision_matrix = robust_cov.precision_;

        #plt.scatter(mahal_robust_cov,mahal_emp_cov);
        #plt.show()

        return cov_matrix, precision_matrix, mahal_robust_cov,pvalue_robust_cov;
    def calculate_EmpiricalCovariance(self,
        params_I={'store_precision':True, 'assume_centered':False}):
        '''
        Maximum likelihood covariance estimator
        sklearn.covariance.EmpiricalCovariance(store_precision=True, assume_centered=False)
        http://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance
        '''
        # compare estimators learned from the full data set with true parameters
        emp_cov = EmpiricalCovariance(**params_I)
        emp_cov.fit(self.data_train['data']);
        mahal_emp_cov = emp_cov.mahalanobis(self.data_train['data']).ravel();
        loglik_emp_cov = emp_cov.score(self.data_train['data'])
        pvalue_emp_cov = np.exp(loglik_emp_cov);
        cov_matrix = emp_cov.covariance_;
        precision_matrix = emp_cov.precision_;

        return cov_matrix, precision_matrix, mahal_emp_cov, pvalue_emp_cov;
