from .calculate_dependencies import *
from .calculate_base import calculate_base

class calculate_importantFeatures(calculate_base):
    def extract_importantFeatures(self,data_model_I=None,raise_I=False):
        '''
        INPUT:
        data_model_I = classification or regression model
        '''
        if data_model_I: data_model=data_model_I;
        else: data_model = self.data_model;
        important_features_O = None;
        try:
            important_features_O = data_model.feature_importances_;
        except Exception as e:
            if raise_I: raise;
            else: print(e);
        return important_features_O;

    def calculate_importantFeatures_std(self,important_features_I,data_model_I=None):
        '''
        calculate the standard deviation of the important features
        based on the feature importances of the estimators
        NOTE: ensemble models only
        INPUT:
        important_features_I = array of important features
        data_model_I = classification or regression model
        OUTPUT:
        n_O = number of estimators
        std_O = standard deviation of the important feature
        '''
        if data_model_I: data_model=data_model_I;
        else: data_model = self.data_model;
        n_O,std_O = np.full(important_features_I.shape,0.),np.full(important_features_I.shape,0.);
        if hasattr(data_model, "estimators_"):
            std_O = np.std([estimator.feature_importances_ for estimator in data_model.estimators_],
                 axis=0);
            n_O = np.full(std_O.shape,len(data_model.estimators_));
        return n_O,std_O;

    def calculate_ZScoreAndPValue(self,value_I,n_I,std_I):
        '''
        calculate the Z-score and p-value
        INPUT:
        value_I = important feature value
        n_I = number of estimators
        std_I = standard deviation of the important feature
        '''
        if not value_I is None:
            zscore_O,pvalue_O=np.full(value_I.shape,0.),np.full(value_I.shape,0.);
        if not n_I is None and not 0.0 in n_I:
            #calculate the standard error
            se_O = std_I/np.sqrt(n_I);
            #calculate the zscore
            if 0.0 in se_O:
                zscore_O=np.full(se_O.shape,1e3); #fill with an arbitrarily large value
                for i in range(se_O.shape[0]):
                    if se_O[i] != 0.0:
                        zscore_O[i] = value_I[i]/se_O[i];
            else:
                zscore_O = value_I/se_O;
            #calculate the pvalue
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

    def extract_dataFeatureSelection_ranking(self,
            data_model_I=None,
            raise_I=False):
        '''
        extract out the ranking from a feature selection data model
        INPUT:
        data_model_I = feature selection model
        '''
        if data_model_I: data_model=data_model_I;
        else: data_model = self.data_featureSelection;
        impfeat_values_O,impfeat_scores_O = None,None
        try:
            impfeat_values_O = data_model.ranking_;
            if hasattr(data_model, "grid_scores_"):
                impfeat_scores_O = data_model.grid_scores_;
            else:
                impfeat_scores_O = np.full(impfeat_values_O.shape,0.)
        except Exception as e:
            if raise_I: raise;
            else: print(e);
        return impfeat_values_O,impfeat_scores_O;

    def extract_coefficientsSVM(self,data_model_I=None,raise_I=False):
        '''
        INPUT:
        data_model_I = support vector machine
        '''
        if data_model_I: data_model=data_model_I;
        else: data_model = self.data_model;
        important_features_O = None;
        try:
            important_features_O = data_model.feature_importances_;
        except Exception as e:
            if raise_I: raise;
            else: print(e);
        return important_features_O;

