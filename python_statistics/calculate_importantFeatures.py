from .calculate_dependencies import *
from .calculate_base import calculate_base

class calculate_importantFeatures(calculate_base):
    def extract_importantFeatures(self,data_model_I=None):
        '''
        INPUT:
        data_model_I = classification or regression model
        '''
        if data_model_I: data_model=data_model_I;
        else: data_model = self.data_model;

        important_features = data_model.feature_importances_;

    def calculate_importantFeatures_std(self,data_model_I=None):
        '''
        calculate the standard deviation of the important features
        based on the feature importances of the estimators
        NOTE: ensemble models only
        INPUT:
        data_model_I = classification or regression model
        OUTPUT:
        n_O = number of estimators
        std_O = standard deviation of the important feature
        '''
        if data_model_I: data_model=data_model_I;
        else: data_model = self.data_model;
        n_O = len(data_model.estimators_);
        std_O = np.std([estimator.feature_importances_ for estimator in data_model.estimators_],
             axis=0);
        return n_O,std_O;

    def calculate_importantFeatures_ZScoreAndPValue(self,value_I,n_I,std_I):
        '''
        calculate the Z-score and p-value for the important feature
        INPUT:
        value_I = important feature value
        n_I = number of estimators
        std_I = standard deviation of the important feature
        '''
        se_O = std_I/np.sqrt(n_I);
        zscore_O = value_I/se_O;
        pvalue_O = scipy.stats.norm.sf(abs(zscore_O));
        return zscore_O,pvalue_O;

    def calculate_importantFeature_jackknife(self,data_model_I=None):
        ''' '''
        pass;

    def calculate_importantFeature_bootstrap(self,data_model_I=None):
        ''' '''
        pass;

    def calculate_VIP(self,data_model_I=None):
        ''' '''
        pass;
    #sklearn.feature_selection.chi2(X, y)	Compute chi-squared stats between each non-negative feature and class.
    #sklearn.feature_selection.f_classif(X, y)	Compute the ANOVA F-value for the provided sample.
    #sklearn.feature_selection.f_regression(X, y[, center])	Univariate linear regression tests.

    def calculate_RFECV(self,data_model_I=None):
        '''
        Recursive feature extraction of linear models
        NOTES: applicable to general linear models and SVM (kernal='linear')
        EXAMPLES:
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        rfe = RFE(clf)
        rfecv = RFECV(clf,cv=10,scoring='accuracy')
        rfe.fit(X_train, y_train)
        rfecv.fit(X_train, y_train)
        rfe.ranking_
        rfecv.ranking_
        rfecv.grid_scores_
        '''
        pass;
