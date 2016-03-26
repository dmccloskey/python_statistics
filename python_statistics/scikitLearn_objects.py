from .scikitLearn_dependencies import *

class scikitLearn_objects():
    def __init__(self):
        self.str2scikitLearnObject_dict = {};
        self.make_str2scikitLearnObject_dict();

    def set_str2scikitLearnObject_dict(self,str2scikitLearnObject_dict_I):
        self.str2scikitLearnObject_dict=str2scikitLearnObject_dict_I;

    def get_str2scikitLearnObject_dict(self):
        return self.str2scikitLearnObject_dict;

    def make_str2scikitLearnObject_dict(self):
        str2scikitLearnObject_dict_I = {
            #Clustering
            'AffinityPropagation':AffinityPropagation, #Perform Affinity Propagation Clustering of data.
            'AgglomerativeClustering':AgglomerativeClustering, #Agglomerative Clustering
            'Birch':Birch, #Implements the Birch clustering algorithm.
            'DBSCAN':DBSCAN, #Perform DBSCAN clustering from vector array or distance matrix.
            'FeatureAgglomeration':FeatureAgglomeration, #Agglomerate features.
            'KMeans':KMeans, #K-Means clustering
            'MiniBatchKMeans':MiniBatchKMeans, #Mini-Batch K-Means clustering
            'MeanShift':MeanShift, #Mean shift clustering using a flat kernel.
            'SpectralClustering':SpectralClustering, #Apply clustering to a projection to the normalized laplacian.

            #Covariance Estimators
            'EmpiricalCovariance':EmpiricalCovariance, #Maximum likelihood covariance estimator
            'EllipticEnvelope':EllipticEnvelope, #An object for detecting outliers in a Gaussian distributed dataset.
            'GraphLasso':GraphLasso, #Sparse inverse covariance estimation with an l1-penalized estimator.
            'GraphLassoCV':GraphLassoCV, #Sparse inverse covariance w/ cross-validated choice of the l1 penalty
            'LedoitWolf':LedoitWolf, #LedoitWolf Estimator
            'MinCovDet':MinCovDet, #Minimum Covariance Determinant MCD): robust estimator of covariance.
            'OAS':OAS, #Oracle Approximating Shrinkage Estimator
            'ShrunkCovariance':ShrunkCovariance, #Covariance estimator with shrinkage
            'empirical_covariance':empirical_covariance, #Computes the Maximum likelihood covariance estimator
            'ledoit_wolf':ledoit_wolf, #Estimates the shrunk Ledoit-Wolf covariance matrix.
            'shrunk_covariance':shrunk_covariance, #Calculates a covariance matrix shrunk on the diagonal
            'oas':oas, #Estimate covariance with the Oracle Approximating Shrinkage algorithm.
            'graph_lasso':graph_lasso, #l1-penalized covariance estimator

            #Cross Validation
            'KFold':KFold, #K-Folds cross validation iterator.
            'LabelKFold':LabelKFold, #K-fold iterator variant with non-overlapping labels.
            'LabelShuffleSplit':LabelShuffleSplit, #Shuffle-Labels-Out cross-validation iterator
            'LeaveOneLabelOut':LeaveOneLabelOut, #Leave-One-Label_Out cross-validation iterator
            'LeaveOneOut':LeaveOneOut, #Leave-One-Out cross validation iterator.
            'LeavePLabelOut':LeavePLabelOut, #Leave-P-Label_Out cross-validation iterator
            'LeavePOut':LeavePOut, #Leave-P-Out cross validation iterator
            'PredefinedSplit':PredefinedSplit, #Predefined split cross validation iterator
            'ShuffleSplit':ShuffleSplit, #Random permutation cross-validation iterator.
            'StratifiedKFold':StratifiedKFold, #Stratified K-Folds cross validation iterator
            'StratifiedShuffleSplit':StratifiedShuffleSplit, #Stratified ShuffleSplit cross validation iterator
            'train_test_split':train_test_split, #Split arrays or matrices into random train and test subsets
            'cross_val_score':cross_val_score, #Evaluate a score by cross-validation
            'cross_val_predict':cross_val_predict, #Generate cross-validated estimates for each input data point
            'permutation_test_score':permutation_test_score, #Evaluate the significance of a cross-validated score with permutations
            'check_cv':check_cv, #Input checker utility for building a CV in a user friendly way.

            #Matrix Decomposition
            'PCA':PCA, #Principal component analysis PCA)
            'IncrementalPCA':IncrementalPCA, #Incremental principal components analysis IPCA).
            'ProjectedGradientNMF':ProjectedGradientNMF, #Non-Negative Matrix Factorization NMF)
            'RandomizedPCA':RandomizedPCA, #Principal component analysis PCA) using randomized SVD
            'KernelPCA':KernelPCA, #Kernel Principal component analysis KPCA)
            'FactorAnalysis':FactorAnalysis, #Factor Analysis FA)
            'FastICA':FastICA, #FastICA: a fast algorithm for Independent Component Analysis.
            'TruncatedSVD':TruncatedSVD, #Dimensionality reduction using truncated SVD aka LSA).
            'NMF':NMF, #Non-Negative Matrix Factorization NMF)
            'SparsePCA':SparsePCA, #Sparse Principal Components Analysis SparsePCA)
            'MiniBatchSparsePCA':MiniBatchSparsePCA, #Mini-batch Sparse Principal Components Analysis
            'SparseCoder':SparseCoder, #Sparse coding
            'DictionaryLearning':DictionaryLearning, #Dictionary learning
            'MiniBatchDictionaryLearning':MiniBatchDictionaryLearning, #Mini-batch dictionary learning
            'LatentDirichletAllocation':LatentDirichletAllocation, #Latent Dirichlet Allocation with online variational Bayes algorithm
            'fastica':fastica, #Perform Fast Independent Component Analysis.
            'dict_learning':dict_learning, #Solves a dictionary learning matrix factorization problem.
            'dict_learning_online':dict_learning_online, #Solves a dictionary learning matrix factorization problem online.
            'sparse_encode':sparse_encode, #Sparse coding

            #Ensemble Methods
            #trees:
            'AdaBoostClassifier':AdaBoostClassifier, #An AdaBoost classifier.
            'RandomForestClassifier':RandomForestClassifier, #A random forest classifier.
            'RandomTreesEmbedding':RandomTreesEmbedding, #An ensemble of totally random trees.
            'RandomForestRegressor':RandomForestRegressor, #A random forest regressor.
            'ExtraTreesClassifier':ExtraTreesClassifier, #An extra-trees classifier.
            'ExtraTreesRegressor':ExtraTreesRegressor, #An extra-trees regressor.

            'AdaBoostRegressor':AdaBoostRegressor, #An AdaBoost regressor.
            'BaggingClassifier':BaggingClassifier, #A Bagging classifier.
            'BaggingRegressor':BaggingRegressor, #A Bagging regressor.
            'GradientBoostingClassifier':GradientBoostingClassifier, #Gradient Boosting for classification.
            'GradientBoostingRegressor':GradientBoostingRegressor, #Gradient Boosting for regression.
            'VotingClassifier':VotingClassifier, #Soft Voting/Majority Rule classifier for unfitted estimators

            #Feature Selection
            'RFE':RFE, #Feature ranking with recursive feature elimination.
            'RFECV':RFECV, #Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
            'chi2':chi2, #Compute chi-squared stats between each non-negative feature and class.
            'f_classif':f_classif, #Compute the ANOVA F-value for the provided sample.
            'f_regression':f_regression, #Univariate linear regression tests.

            #Gaussian Processes
            'GaussianProcess':GaussianProcess, #The Gaussian Process model class.

            #Grid Search
            'GridSearchCV':GridSearchCV, #Exhaustive search over specified parameter values for an estimator.
            'RandomizedSearchCV':RandomizedSearchCV, #Randomized search on hyper parameters.


            #Kernel Ridge Regression
            'KernelRidge':KernelRidge, #Kernel ridge regression.

            #Discriminant Analysis
            'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis, #Linear Discriminant Analysis
            'QuadraticDiscriminantAnalysis':QuadraticDiscriminantAnalysis, #Quadratic Discriminant Analysis


            #Generalized Linear Models
            'ARDRegression':ARDRegression, #Bayesian ARD regression.
            'BayesianRidge':BayesianRidge, #Bayesian ridge regression
            'ElasticNet':ElasticNet, #Linear regression with combined L1 and L2 priors as regularizer.
            'ElasticNetCV':ElasticNetCV, #Elastic Net model with iterative fitting along a regularization path
            'Lars':Lars, #Least Angle Regression model a from k from a.
            'LarsCV':LarsCV, #Cross-validated Least Angle Regression model
            'Lasso':Lasso, #Linear Model trained with L1 prior as regularizer aka the Lasso)
            'LassoCV':LassoCV, #Lasso linear model with iterative fitting along a regularization path
            'LassoLars':LassoLars, #Lasso model fit with Least Angle Regression a from k from a.
            'LassoLarsCV':LassoLarsCV, #Cross-validated Lasso, using the LARS algorithm
            'LassoLarsIC':LassoLarsIC, #Lasso model fit with Lars using BIC or AIC for model selection
            'LinearRegression':LinearRegression, #Ordinary least squares Linear Regression.
            'LogisticRegression':LogisticRegression, #Logistic Regression aka logit, MaxEnt) classifier.
            'LogisticRegressionCV':LogisticRegressionCV, #Logistic Regression CV aka logit, MaxEnt) classifier.
            'MultiTaskLasso':MultiTaskLasso, #Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer
            'MultiTaskElasticNet':MultiTaskElasticNet, #Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer
            'MultiTaskLassoCV':MultiTaskLassoCV, #Multi-task L1/L2 Lasso with built-in cross-validation.
            'MultiTaskElasticNetCV':MultiTaskElasticNetCV, #Multi-task L1/L2 ElasticNet with built-in cross-validation.
            'OrthogonalMatchingPursuit':OrthogonalMatchingPursuit, #Orthogonal Matching Pursuit model OMP)
            'OrthogonalMatchingPursuitCV':OrthogonalMatchingPursuitCV, #Cross-validated Orthogonal Matching Pursuit model OMP)
            'PassiveAggressiveClassifier':PassiveAggressiveClassifier, #Passive Aggressive Classifier
            'PassiveAggressiveRegressor':PassiveAggressiveRegressor, #Passive Aggressive Regressor
            'Perceptron':Perceptron, #Read more in the User Guide.
            'RandomizedLasso':RandomizedLasso, #Randomized Lasso.
            'RandomizedLogisticRegression':RandomizedLogisticRegression, #Randomized Logistic Regression
            'RANSACRegressor':RANSACRegressor, #RANSAC RANdom SAmple Consensus) algorithm.
            'Ridge':Ridge, #Linear least squares with l2 regularization.
            'RidgeClassifier':RidgeClassifier, #Classifier using Ridge regression.
            'RidgeClassifierCV':RidgeClassifierCV, #Ridge classifier with built-in cross-validation.
            'RidgeCV':RidgeCV, #Ridge regression with built-in cross-validation.
            'SGDClassifier':SGDClassifier, #Linear classifiers SVM, logistic regression, a from o from ) with SGD training.
            'SGDRegressor':SGDRegressor, #Linear model fitted by minimizing a regularized empirical loss with SGD
            'TheilSenRegressor':TheilSenRegressor, #Theil-Sen Estimator: robust multivariate regression model.
            'lars_path':lars_path, #Compute Least Angle Regression or Lasso path using LARS algorithm [1]
            'lasso_path':lasso_path, #Compute Lasso path with coordinate descent
            'lasso_stability_path':lasso_stability_path, #Stabiliy path based on randomized Lasso estimates
            'orthogonal_mp':orthogonal_mp, #Orthogonal Matching Pursuit OMP)
            'orthogonal_mp_gram':orthogonal_mp_gram, #Gram Orthogonal Matching Pursuit OMP)

            #Manifold Learning
            'LocallyLinearEmbedding':LocallyLinearEmbedding, #Locally Linear Embedding
            'Isomap':Isomap, #Isomap Embedding
            'MDS':MDS, #Multidimensional scaling
            'SpectralEmbedding':SpectralEmbedding, #Spectral embedding for non-linear dimensionality reduction.
            'TSNE':TSNE, #t-distributed Stochastic Neighbor Embedding.
            'locally_linear_embedding':locally_linear_embedding, #Perform a Locally Linear Embedding analysis on the data.
            'spectral_embedding':spectral_embedding, #Project the sample on the first eigenvectors of the graph Laplacian.

            #Metrics
            #Classification metrics
            'accuracy_score':accuracy_score, #Accuracy classification score.
            'auc':auc, #Compute Area Under the Curve AUC) using the trapezoidal rule
            'average_precision_score':average_precision_score, #Compute average precision AP) from prediction scores
            'brier_score_loss':brier_score_loss, #Compute the Brier score.
            'classification_report':classification_report, #Build a text report showing the main classification metrics
            'confusion_matrix':confusion_matrix, #Compute confusion matrix to evaluate the accuracy of a classification
            'f1_score':f1_score, #Compute the F1 score, also known as balanced F-score or F-measure
            'fbeta_score':fbeta_score, #Compute the F-beta score
            'hamming_loss':hamming_loss, #Compute the average Hamming loss.
            'hinge_loss':hinge_loss, #Average hinge loss non-regularized)
            'jaccard_similarity_score':jaccard_similarity_score, #Jaccard similarity coefficient score
            'log_loss':log_loss, #Log loss, aka logistic loss or cross-entropy loss.
            'matthews_corrcoef':matthews_corrcoef, #Compute the Matthews correlation coefficient MCC) for binary classes
            'precision_recall_curve':precision_recall_curve, #Compute precision-recall pairs for different probability thresholds
            'precision_recall_fscore_support':precision_recall_fscore_support, #Compute precision, recall, F-measure and support for each class
            'precision_score':precision_score, #Compute the precision
            'recall_score':recall_score, #Compute the recall
            'roc_auc_score':roc_auc_score, #Compute Area Under the Curve AUC) from prediction scores
            'roc_curve':roc_curve, #Compute Receiver operating characteristic ROC)
            'zero_one_loss':zero_one_loss, #Zero-one classification loss.
            'brier_score_loss':brier_score_loss, #Compute the Brier score.
            #Regression metrics
            'explained_variance_score':explained_variance_score, #Explained variance regression score function
            'mean_absolute_error':mean_absolute_error, #Mean absolute error regression loss
            'mean_squared_error':mean_squared_error, #Mean squared error regression loss
            'median_absolute_error':median_absolute_error, #Median absolute error regression loss
            'r2_score':r2_score, #R^2 coefficient of determination) regression score function.
            #Clustering metrics
            'adjusted_mutual_info_score':adjusted_mutual_info_score, #Adjusted Mutual Information between two clusterings
            'adjusted_rand_score':adjusted_rand_score, #Rand index adjusted for chance
            'completeness_score':completeness_score, #Completeness metric of a cluster labeling given a ground truth
            'homogeneity_completeness_v_measure':homogeneity_completeness_v_measure, #Compute the homogeneity and completeness and V-Measure scores at once
            'homogeneity_score':homogeneity_score, #Homogeneity metric of a cluster labeling given a ground truth
            'mutual_info_score':mutual_info_score, #Mutual Information between two clusterings
            'normalized_mutual_info_score':normalized_mutual_info_score, #Normalized Mutual Information between two clusterings
            'silhouette_score':silhouette_score, #Compute the mean Silhouette Coefficient of all samples.
            'silhouette_samples':silhouette_samples, #Compute the Silhouette Coefficient for each sample.
            'v_measure_score':v_measure_score, #V-measure cluster labeling given a ground truth.
            #Pairwise metrics
            'additive_chi2_kernel':additive_chi2_kernel, #Computes the additive chi-squared kernel between observations in X and Y
            'chi2_kernel':chi2_kernel, #Computes the exponential chi-squared kernel X and Y.
            'distance_metrics':distance_metrics, #Valid metrics for pairwise_distances.
            'euclidean_distances':euclidean_distances, #Considering the rows of X and Y=X) as vectors, compute the distance matrix between each pair of vectors.
            'kernel_metrics':kernel_metrics, #Valid metrics for pairwise_kernels
            'linear_kernel':linear_kernel, #Compute the linear kernel between X and Y.
            'manhattan_distances':manhattan_distances, #Compute the L1 distances between the vectors in X and Y.
            'pairwise_distances':pairwise_distances, #Compute the distance matrix from a vector array X and optional Y.
            'pairwise_kernels':pairwise_kernels, #Compute the kernel between arrays X and optional array Y.
            'polynomial_kernel':polynomial_kernel, #Compute the polynomial kernel between X and Y:
            'rbf_kernel':rbf_kernel, #Compute the rbf gaussian) kernel between X and Y:
            'laplacian_kernel':laplacian_kernel, #Compute the laplacian kernel between X and Y.
            'pairwise_distances':pairwise_distances, #Compute the distance matrix from a vector array X and optional Y.
            'pairwise_distances_argmin':pairwise_distances_argmin, #Compute minimum distances between one point and a set of points.
            'pairwise_distances_argmin_min':pairwise_distances_argmin_min, #Compute minimum distances between one point and a set of points.

            #Naive Bayes
            'GaussianNB':GaussianNB, #Gaussian Naive Bayes (GaussianNB)
            'MultinomialNB':MultinomialNB, #Naive Bayes classifier for multinomial models
            'BernoulliNB':BernoulliNB, #Naive Bayes classifier for multivariate Bernoulli models.

            #Nearest Neighbors
            'NearestNeighbors':NearestNeighbors, #Unsupervised learner for implementing neighbor searches.
            'KNeighborsClassifier':KNeighborsClassifier, #Classifier implementing the k-nearest neighbors vote.
            'RadiusNeighborsClassifier':RadiusNeighborsClassifier, #Classifier implementing a vote among neighbors within a given radius
            'KNeighborsRegressor':KNeighborsRegressor, #Regression based on k-nearest neighbors.
            'RadiusNeighborsRegressor':RadiusNeighborsRegressor, #Regression based on neighbors within a fixed radius.
            'NearestCentroid':NearestCentroid, #Nearest centroid classifier.
            'BallTree':BallTree,
            'KDTree':KDTree,
            'LSHForest':LSHForest, #Performs approximate nearest neighbor search using LSH forest.
            'DistanceMetric':DistanceMetric,
            'KernelDensity':KernelDensity, #Kernel Density Estimation
            'kneighbors_graph':kneighbors_graph, #Computes the weighted) graph of k-Neighbors for points in X
            'radius_neighbors_graph':radius_neighbors_graph, #Computes the weighted) graph of Neighbors for points in X

            ##Neural network models
            #'GoogLeNetClassifier':GoogLeNetClassifier,
            #'OverfeatClassifier':OverfeatClassifier,

            #Cross decomposition
            'PLSRegression':PLSRegression, #PLS regression
            'PLSCanonical':PLSCanonical, #PLSCanonical implements the 2 blocks canonical PLS of the original Wold algorithm [Tenenhaus 1998] p from 204, referred as PLS-C2A in [Wegelin 2000].
            'CCA':CCA, #CCA Canonical Correlation Analysis.
            'PLSSVD':PLSSVD, #Partial Least Square SVD

            #Pipeline
            'Pipeline':Pipeline, #Pipeline of transforms with a final estimator.
            'FeatureUnion':FeatureUnion, #Concatenates results of multiple transformer objects.

            #Preprocessing and Normalization

            'scale':scale, #Standardize a dataset along any axis

            #Support Vector Machines
            'SVC':SVC, #C-Support Vector Classification.
            'LinearSVC':LinearSVC, #Linear Support Vector Classification.
            'NuSVC':NuSVC, #Nu-Support Vector Classification.
            'SVR':SVR, #Epsilon-Support Vector Regression.
            'LinearSVR':LinearSVR, #Linear Support Vector Regression.
            'NuSVR':NuSVR, #Nu Support Vector Regression.
            'OneClassSVM':OneClassSVM, #Unsupervised Outlier Detection.
            'l1_min_c':l1_min_c, #Return the lowest bound for C such that for C in l1_min_C, infinity) the model is guaranteed not to be empty.

            #Decision Trees
            'DecisionTreeClassifier':DecisionTreeClassifier, #A decision tree classifier.
            'DecisionTreeRegressor':DecisionTreeRegressor, #A decision tree regressor.
            'ExtraTreeClassifier':ExtraTreeClassifier, #An extremely randomized tree classifier.
            'ExtraTreeRegressor':ExtraTreeRegressor, #An extremely randomized tree regressor.
            'export_graphviz':export_graphviz, #Export a decision tree in DOT format from 
            };
        self.str2scikitLearnObject_dict=str2scikitLearnObject_dict_I;
    def get_scikitLearnObjectFromStr2scikitLearnObjectDict(self,object_I):
        '''get the scikit-learn object by name'''
        scikitLearnObject_O = None;
        if object_I in self.str2scikitLearnObject_dict.keys():
            scikitLearnObject_O = self.str2scikitLearnObject_dict[object_I];
        else:
            print('object not found.');
        return scikitLearnObject_O;