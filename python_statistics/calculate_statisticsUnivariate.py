from .calculate_dependencies import *
from .calculate_base import calculate_base
class calculate_statisticsUnivariate(calculate_base):

    def calculate_pValueCorrected(self,pvalue_I,alpha,method):
        '''calculate the corrected p-value
        INPUT:
        pvalue_I = float, uncorrected p-value
        alpha = float,
        method = string, method name
        OUTPUT:
        pvalue_corrected_O = float, corrected p-value
        TODO:
        http://jpktd.blogspot.dk/2013/04/multiple-testing-p-value-corrections-in.html
        OR use p.adjust in rpy2
        '''
        pvalue_corrected_O = pvalue_I;
        #adjust the p-value
        return pvalue_corrected_O;

    def calculate_anova(self):
        '''Perform ANOVA
        scipy.stats.f_oneway(*args)
        scipy.stats.f
        '''
        pass;

    def calculate_oneSampleTTest(self,data_1,pop_mean_I=0.0):
        '''Calculate the 1-sample t-test on a data set
        INPUT:
        data_1 = array of floats
        pop_mean_I = population mean, default = 0.0
        OUTPUT:
        tstat_O
        pval_O
        '''
        tstat_O,pval_O = scipy.stats.ttest_1samp(data_1,pop_mean_I);
        return tstat_O,pval_O;

    def calculate_twoSampleTTest(self,data_1,data_2,equal_var_I = False):
        '''Calculate the independent 2-way t-test on two data sets
        INPUT:
        data_1 = array of floats
        data_2 = array of floats
        equal_var_I = Boolean, assume the variance of the data sets are equal or not
        OUTPUT:
        tstat_O
        pval_O
        '''
        tstat_O,pval_O = scipy.stats.ttest_ind(data_1,data_2, equal_var = equal_var_I);
        return tstat_O,pval_O;

    def calculate_pairwiseTTest(self,data_1,data_2):
        '''Calculate the pairwise 2-way t-test on two data sets
        INPUT:
        data_1 = array of floats
        data_2 = array of floats
        OUTPUT:
        tstat_O
        pval_O
        '''
        tstat_O,pval_O = scipy.stats.ttest_rel(data_1,data_2);
        return tstat_O,pval_O;

    def calculate_twoSampleTTest_descriptiveStats(self,
                    data_1_mean,data_1_var,
                    data_2_mean,data_2_var):
        '''Calculate the independent 2-way t-test on two data sets
        using the descriptive stats
        INPUT:
        data_1_mean = float, mean
        data_1_var = float, variance
        data_2_mean = float, mean
        data_2_var = float, variance
        OUTPUT:
        tstat_O
        pval_O
        '''
        from scipy.special import stdtr

        # calculate the size and dofs
        ndata_1 = len(data_1_mean);
        data_1_dof = ndata_1-1;
        ndata_2 = len(data_2_mean);
        data_2_dof = ndata_2-1;

        # calculate Welch's t-test using the descriptive statistics.
        tstat_O = (data_1_mean - data_2_mean) / np.sqrt(data_1_var/ndata_1 + data_2_var/ndata_2);
        dof = (data_1_var/ndata_1+ data_2_var/ndata_2)**2 / (data_1_var**2/(ndata_1**2*data_1_dof) + data_2_var**2/(ndata_2**2*data_2_dof));
        pval_O = 2*stdtr(dof, -np.abs(tf));
        return tstat_O,pval_O;

    def calculate_oneSampleTTest_descriptiveStats(self,
                    data_1_mean,data_1_var,pop_mean_I=0.0):
        '''Calculate the independent 2-way t-test on one data set
        using the descriptive stats
        INPUT:
        data_1_mean = float, mean
        data_1_var = float, variance
        pop_mean_I = population mean, default = 0.0
        OUTPUT:
        tstat_O
        pval_O
        '''
        from scipy.special import stdtr

        # calculate the size and dofs
        ndata_1 = len(data_1_mean);
        data_1_dof = ndata_1-2;

        # calculate Welch's t-test using the descriptive statistics.
        tstat_O = (data_1_mean - pop_mean_I) / np.sqrt(data_1_var/ndata_1);
        pval_O = 2*stdtr(data_1_dof, -np.abs(tf));
        return tstat_O,pval_O;
