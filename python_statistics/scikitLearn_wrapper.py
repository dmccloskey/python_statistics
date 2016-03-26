from .calculate_dependencies import *
from .calculate_base import calculate_base
from .scikitLearn_objects import scikitLearn_objects

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
            if 'response' in self.data_train.keys():
                data_model.fit(self.data_train['data'],self.data_train['response']);
            else:
                data_model.fit(self.data_train['data']);
            self.data_model = data_model;
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    def make_dataFeatureSelection(self,
            model_I,model_parameters_I,
            impfeat_method_I,impfeat_options_I,
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
            model_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(model_I);
            estimator = model_obj(**model_parameters_I);
            rfecv_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(impfeat_method_I);
            data_model = rfecv_obj(estimator,**impfeat_options_I);
            if 'response' in self.data_train.keys():
                data_model.fit(self.data_train['data'],self.data_train['response']);
            else:
                data_model.fit(self.data_train['data']);
            self.data_featureSelection = data_model;
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    #TODO:
    def make_dataModelAndHyperParameterCV(self,
            model_I,parameters_I,
            hyperParameter_method_I,hyperParameter_options_I,
            crossVal_method_I=None,crossVal_options_I=None,
            score_method_I=None,score_options_I=None,
            raise_I=False):
        '''
        call a scikit-learn estimator, and fit the data
        INPUT:
        OUTPUT:
        '''
        try:
            scikitLearn_obj = scikitLearn_objects();
            model_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(model_I);
            estimator = model_obj(**parameters_I);
            hyperParameter_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(hyperParameter_method_I);
            data_model = hyperParameter_obj(estimator,**crossVal_options_I);
            if 'response' in self.data_train.keys():
                data_model.fit(self.data_train['data'],self.data_train['response']);
            else:
                data_model.fit(self.data_train['data']);
            self.data_model = data_model;
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    def make_dataPipeline(self,
            models_I,
            raise_I=False):
        '''
        call a scikit-learn pipeline, and fit the data
        INPUT:
        OUTPUT:
        '''
        try:
            scikitLearn_obj = scikitLearn_objects();
            pipeline = [];
            for model in models_I:
                model_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(model['model']);
                estimator = model_obj(**model['parameters']);
                pipeline.append((model['model'],estimator));
            data_model = Pipeline(pipeline);
            if 'response' in self.data_train.keys():
                data_model.fit(self.data_train['data'],self.data_train['response']);
            else:
                data_model.fit(self.data_train['data']);
            self.data_model = data_model;
        except Exception as e:
            if raise_I: raise;
            else: print(e);
