from .calculate_dependencies import *
from .calculate_base import calculate_base

from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import permutation_test_score
from sklearn.metrics import confusion_matrix,\
        accuracy_score,precision_score,recall_score,\
        roc_auc_score

class calculate_modelValidation(calculate_base):
    
    def model_crossValidate(self,X,y,classifier,cv):
        '''cross validate the model
        METHOD:
        cross_val_score uses StratifiedKFold by default
        other cross_validation classes include KFold, StratifiedKFold, ShuffleSplit, LeavePLabelOut
        INPUT:
        E.g.
        from sklearn.datasets import load_iris
        from sklearn.neighbors import KNeighborsClassifier

        iris = load_iris()
        X, y = iris.data, iris.target

        classifier = KNeighborsClassifier()

        X = samples, array of shape len(n_samples),len(n_features)
        y = target, array of shape len(n_samples)
        classifier = class, classifier used
        cv = number of folds or object of class cross_validation

        where cv
        cv = KFold(n=len(x), shuffle=True)
        cv = StratifiedKFold(iris.target, n_folds=5)
        cv = ShuffleSplit(len(iris.target), n_iter=5, test_size=.2)
        ...
        
        '''        

        scores = cross_val_score(classifier, X, y,cv=cv);

        pass;

    def model_gridSearch_parameters(self):
        '''perform a grid serch of the parameter space
        E.g.:
        from sklearn.grid_search import GridSearchCV
        from sklearn.svm import SVR
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}

        grid = GridSearchCV(SVR(), param_grid=param_grid, cv=cv, verbose=3)
        grid.fit(X, y)
        grid.predict(X)
        print(grid.best_score_)
        print(grid.best_params_)

        if over fitting is an issue
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
        cv = KFold(n=len(X_train), n_folds=10, shuffle=True)

        grid = GridSearchCV(SVR(), param_grid=param_grid, cv=cv)

        grid.fit(X_train, y_train)
        grid.score(X_test, y_test)
        '''

    def model_permute(self):
        '''perform a permutation test on a cross-validated model to test for significance

        from sklearn.cross_validation import permutation_test_score
        
        '''
        pass;

    def score_model(model):
        '''
        sklearn.metrics.make_scorer(score_func, greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs)[source]

        EXAMPLE:
        https://www.kaggle.com/srodriguex/santander-customer-satisfaction/model-and-feature-selection-with-python/notebook
        skf = cv.StratifiedKFold(y, n_folds=3, shuffle=True)
        score_metric = 'roc_auc'
        scores = {}
        return cv.cross_val_score(model, X, y, cv=skf, scoring=score_metric)

        '''
        
        from sklearn.cross_validation import cross_val_score
        scores = cross_val_score(logreg, features_array, target, cv=5,
                         scoring='roc_auc')
        scores.min(), scores.mean(), scores.max()
    def make_modelScorer(model):
        '''
        sklearn.metrics.make_scorer(score_func, greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs)[source]
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
        EXAMPLE:
        from sklearn.metrics import fbeta_score, make_scorer
        ftwo_scorer = make_scorer(fbeta_score, beta=2)
        '''

    def calculate_classificationMetrics(model):
        '''
        '''
        confusion_matrix(y, y_pred)
        accuracy_score(y, y_pred)
        precision_score(y, y_pred)
        recall_score(y, y_pred)

        def plot_confusion(cm, target_names = ['survived', 'not survived'],
                   title='Confusion matrix'):
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(title)
            plt.colorbar()

            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=60)
            plt.yticks(tick_marks, target_names)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            # Convenience function to adjust plot parameters for a clear layout.
            plt.tight_layout()
    
        plot_confusion(cm)