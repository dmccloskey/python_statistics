from .calculate_dependencies import *
from .calculate_base import calculate_base
from .scikitLearn_objects import scikitLearn_objects
from sklearn.pipeline import Pipeline

class scikitLearn_wrapper(calculate_base):
    def make_dataModel(self,model_I,parameters_I,raise_I=False):
        '''
        call a scikit-learn estimator and fit the data
        INPUT:
        OUTPUT:
        '''
        try:
            scikitLearn_obj = scikitLearn_objects();
            model_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(model_I);
            data_model = model_obj(**parameters_I);
            self.data_model = data_model;
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    def make_dataFeatureSelection(self,
            impfeat_method_I,impfeat_options_I,
            data_model_I=None,
            raise_I=False):
        '''
        Recursive feature extraction of linear models
        NOTES: applicable to general linear models, trees, and SVM (kernal='linear')
        EXAMPLES:
        RFECV(estimator, step=1, cv=None, scoring=None, estimator_params=None, verbose=0)
        RFE(estimator, n_features_to_select=None, step=1, estimator_params=None, verbose=0)
        '''
        try:
            scikitLearn_obj = scikitLearn_objects();
            # get the estimator
            if data_model_I: estimator=data_model_I;
            else: estimator = self.data_model;  
            estimator = self.get_finalEstimator(estimator);

            rfecv_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(impfeat_method_I);
            data_model = rfecv_obj(estimator,**impfeat_options_I);
            self.data_featureSelection = data_model;
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    def make_dataHyperparameterCV(self,
            param_dist_I,
            hyperparameter_method_I,hyperparameter_options_I,
            data_model_I=None,
            crossval_method_I=None,crossval_options_I=None,crossval_labels_I=None,
            metric_method_I=None,metric_options_I=None,
            raise_I=False):
        '''
        call a scikit-learn estimator, and fit the data
        INPUT:
        OUTPUT:
        '''
        scikitLearn_obj = scikitLearn_objects(); 
        # get the estimator
        if data_model_I: estimator=data_model_I;
        else: estimator = self.data_model;  
        try:         
            # get the scoring metric
            if metric_method_I is None:
                scorer = None;
            elif type(metric_method_I)==type(''):
                scorer = metric_method_I;
            else:
                score_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(metric_method_I);
                scorer = score_obj(**metric_options_I);
            # get the CV method  
            if type(crossval_method_I) == type(1):
                validator = crossval_method_I;
            else:
                cv_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(crossval_method_I);
                validator = cv_obj(crossval_labels_I,**crossval_options_I);
            # make the hyper parameter search method
            hyperparameter_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(hyperparameter_method_I);
            if "Grid" in hyperparameter_method_I:
                data_model = hyperparameter_obj(
                    estimator,
                    param_grid=param_dist_I,
                    cv=validator,
                    scoring=scorer,
                    **hyperparameter_options_I);
            elif "Randomized" in hyperparameter_method_I:
                data_model = hyperparameter_obj(
                    estimator,
                    param_distributions=param_dist_I,
                    cv=validator,
                    scoring=scorer,
                    **hyperparameter_options_I);
            self.data_hyperparameterCV = data_model;
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    def make_dataPipeline(self,
            models_I,parameters_I,
            raise_I=False):
        '''
        call a scikit-learn pipeline, and fit the data
        INPUT:
        OUTPUT:
        '''
        try:
            scikitLearn_obj = scikitLearn_objects();
            pipeline = [];
            connector_str = '__'
            pipeline_parameters={};
            for i,model in enumerate(models_I):
                model_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(model);
                if not parameters_I[i] is None:
                    model_connector_str = '%s%s' %(model,connector_str);
                    tmp = {model_connector_str+k:v for k,v in parameters_I[i].items()};
                    pipeline_parameters.update(tmp);
                estimator = model_obj();
                pipeline.append((model,estimator));
            data_model = Pipeline(pipeline);
            data_model.set_params(**pipeline_parameters);
            self.data_model = data_model;
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    def fit_data2Model(self,
            raise_I=False):
        '''
        call a scikit-learn pipeline, and fit the data
        INPUT:
        OUTPUT:
        '''
        try:
            if 'response' in self.data_train.keys():
                self.data_model.fit(self.data_train['data'],self.data_train['response']);
            else:
                self.data_model.fit(self.data_train['data']);
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    def fit_data2HyperparameterCV(self,
            raise_I=False):
        '''
        call a scikit-learn pipeline, and fit the data
        INPUT:
        OUTPUT:
        '''
        try:
            if 'response' in self.data_train.keys():
                self.data_hyperparameterCV.fit(self.data_train['data'],self.data_train['response']);
            else:
                self.data_hyperparameterCV.fit(self.data_train['data']);
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    def fit_data2FeatureSelection(self,
            raise_I=False):
        '''
        call a scikit-learn pipeline, and fit the data
        INPUT:
        OUTPUT:
        '''
        try:
            if 'response' in self.data_train.keys():
                self.data_featureSelection.fit(self.data_train['data'],self.data_train['response']);
            else:
                self.data_featureSelection.fit(self.data_train['data']);
        except Exception as e:
            if raise_I: raise;
            else: print(e);