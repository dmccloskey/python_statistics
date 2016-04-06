#Clustering
from sklearn.cluster import AffinityPropagation #Perform Affinity Propagation Clustering of data.
from sklearn.cluster import AgglomerativeClustering #Agglomerative Clustering
from sklearn.cluster import Birch #Implements the Birch clustering algorithm.
from sklearn.cluster import DBSCAN #Perform DBSCAN clustering from vector array or distance matrix.
from sklearn.cluster import FeatureAgglomeration #Agglomerate features.
from sklearn.cluster import KMeans #K-Means clustering
from sklearn.cluster import MiniBatchKMeans #Mini-Batch K-Means clustering
from sklearn.cluster import MeanShift #Mean shift clustering using a flat kernel.
from sklearn.cluster import SpectralClustering #Apply clustering to a projection to the normalized laplacian.

#Covariance Estimators
from sklearn.covariance import EmpiricalCovariance #Maximum likelihood covariance estimator
from sklearn.covariance import EllipticEnvelope #An object for detecting outliers in a Gaussian distributed dataset.
from sklearn.covariance import GraphLasso #Sparse inverse covariance estimation with an l1-penalized estimator.
from sklearn.covariance import GraphLassoCV #Sparse inverse covariance w/ cross-validated choice of the l1 penalty
from sklearn.covariance import LedoitWolf #LedoitWolf Estimator
from sklearn.covariance import MinCovDet #Minimum Covariance Determinant MCD): robust estimator of covariance.
from sklearn.covariance import OAS #Oracle Approximating Shrinkage Estimator
from sklearn.covariance import ShrunkCovariance #Covariance estimator with shrinkage
from sklearn.covariance import empirical_covariance #Computes the Maximum likelihood covariance estimator
from sklearn.covariance import ledoit_wolf #Estimates the shrunk Ledoit-Wolf covariance matrix.
from sklearn.covariance import shrunk_covariance #Calculates a covariance matrix shrunk on the diagonal
from sklearn.covariance import oas #Estimate covariance with the Oracle Approximating Shrinkage algorithm.
from sklearn.covariance import graph_lasso #l1-penalized covariance estimator

#Cross Validation
from sklearn.cross_validation import KFold #K-Folds cross validation iterator.
from sklearn.cross_validation import LabelKFold #K-fold iterator variant with non-overlapping labels.
from sklearn.cross_validation import LabelShuffleSplit #Shuffle-Labels-Out cross-validation iterator
from sklearn.cross_validation import LeaveOneLabelOut #Leave-One-Label_Out cross-validation iterator
from sklearn.cross_validation import LeaveOneOut #Leave-One-Out cross validation iterator.
from sklearn.cross_validation import LeavePLabelOut #Leave-P-Label_Out cross-validation iterator
from sklearn.cross_validation import LeavePOut #Leave-P-Out cross validation iterator
from sklearn.cross_validation import PredefinedSplit #Predefined split cross validation iterator
from sklearn.cross_validation import ShuffleSplit #Random permutation cross-validation iterator.
from sklearn.cross_validation import StratifiedKFold #Stratified K-Folds cross validation iterator
from sklearn.cross_validation import StratifiedShuffleSplit #Stratified ShuffleSplit cross validation iterator
from sklearn.cross_validation import train_test_split #Split arrays or matrices into random train and test subsets
from sklearn.cross_validation import cross_val_score #Evaluate a score by cross-validation
from sklearn.cross_validation import cross_val_predict #Generate cross-validated estimates for each input data point
from sklearn.cross_validation import permutation_test_score #Evaluate the significance of a cross-validated score with permutations
from sklearn.cross_validation import check_cv #Input checker utility for building a CV in a user friendly way.

#Matrix Decomposition
from sklearn.decomposition import PCA #Principal component analysis PCA)
from sklearn.decomposition import IncrementalPCA #Incremental principal components analysis IPCA).
from sklearn.decomposition import ProjectedGradientNMF #Non-Negative Matrix Factorization NMF)
from sklearn.decomposition import RandomizedPCA #Principal component analysis PCA) using randomized SVD
from sklearn.decomposition import KernelPCA #Kernel Principal component analysis KPCA)
from sklearn.decomposition import FactorAnalysis #Factor Analysis FA)
from sklearn.decomposition import FastICA #FastICA: a fast algorithm for Independent Component Analysis.
from sklearn.decomposition import TruncatedSVD #Dimensionality reduction using truncated SVD aka LSA).
from sklearn.decomposition import NMF #Non-Negative Matrix Factorization NMF)
from sklearn.decomposition import SparsePCA #Sparse Principal Components Analysis SparsePCA)
from sklearn.decomposition import MiniBatchSparsePCA #Mini-batch Sparse Principal Components Analysis
from sklearn.decomposition import SparseCoder #Sparse coding
from sklearn.decomposition import DictionaryLearning #Dictionary learning
from sklearn.decomposition import MiniBatchDictionaryLearning #Mini-batch dictionary learning
from sklearn.decomposition import LatentDirichletAllocation #Latent Dirichlet Allocation with online variational Bayes algorithm
from sklearn.decomposition import fastica #Perform Fast Independent Component Analysis.
from sklearn.decomposition import dict_learning #Solves a dictionary learning matrix factorization problem.
from sklearn.decomposition import dict_learning_online #Solves a dictionary learning matrix factorization problem online.
from sklearn.decomposition import sparse_encode #Sparse coding

#Ensemble Methods
#trees:
from sklearn.ensemble import AdaBoostClassifier #An AdaBoost classifier.
from sklearn.ensemble import RandomForestClassifier #A random forest classifier.
from sklearn.ensemble import RandomTreesEmbedding #An ensemble of totally random trees.
from sklearn.ensemble import RandomForestRegressor #A random forest regressor.
from sklearn.ensemble import ExtraTreesClassifier #An extra-trees classifier.
from sklearn.ensemble import ExtraTreesRegressor #An extra-trees regressor.

from sklearn.ensemble import AdaBoostRegressor #An AdaBoost regressor.
from sklearn.ensemble import BaggingClassifier #A Bagging classifier.
from sklearn.ensemble import BaggingRegressor #A Bagging regressor.
from sklearn.ensemble import GradientBoostingClassifier #Gradient Boosting for classification.
from sklearn.ensemble import GradientBoostingRegressor #Gradient Boosting for regression.
from sklearn.ensemble import VotingClassifier #Soft Voting/Majority Rule classifier for unfitted estimators

#Feature Selection
from sklearn.feature_selection import RFE #Feature ranking with recursive feature elimination.
from sklearn.feature_selection import RFECV #Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
from sklearn.feature_selection import chi2 #Compute chi-squared stats between each non-negative feature and class.
from sklearn.feature_selection import f_classif #Compute the ANOVA F-value for the provided sample.
from sklearn.feature_selection import f_regression #Univariate linear regression tests.

#Gaussian Processes
#NOTES: perhaps useful when specifying the "nugget"
from sklearn.gaussian_process import GaussianProcess #The Gaussian Process model class.

#Grid Search
from sklearn.grid_search import GridSearchCV #Exhaustive search over specified parameter values for an estimator.
from sklearn.grid_search import RandomizedSearchCV #Randomized search on hyper parameters.


#Kernel Ridge Regression
#NOTES: alternative to SVR
from sklearn.kernel_ridge import KernelRidge #Kernel ridge regression.

#Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #Linear Discriminant Analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis #Quadratic Discriminant Analysis


#Generalized Linear Models
from sklearn.linear_model import ARDRegression #Bayesian ARD regression.
from sklearn.linear_model import BayesianRidge #Bayesian ridge regression
from sklearn.linear_model import ElasticNet #Linear regression with combined L1 and L2 priors as regularizer.
from sklearn.linear_model import ElasticNetCV #Elastic Net model with iterative fitting along a regularization path
from sklearn.linear_model import Lars #Least Angle Regression model a from k from a.
from sklearn.linear_model import LarsCV #Cross-validated Least Angle Regression model
from sklearn.linear_model import Lasso #Linear Model trained with L1 prior as regularizer aka the Lasso)
from sklearn.linear_model import LassoCV #Lasso linear model with iterative fitting along a regularization path
from sklearn.linear_model import LassoLars #Lasso model fit with Least Angle Regression a from k from a.
from sklearn.linear_model import LassoLarsCV #Cross-validated Lasso, using the LARS algorithm
from sklearn.linear_model import LassoLarsIC #Lasso model fit with Lars using BIC or AIC for model selection
from sklearn.linear_model import LinearRegression #Ordinary least squares Linear Regression.
from sklearn.linear_model import LogisticRegression #Logistic Regression aka logit, MaxEnt) classifier.
from sklearn.linear_model import LogisticRegressionCV #Logistic Regression CV aka logit, MaxEnt) classifier.
from sklearn.linear_model import MultiTaskLasso #Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer
from sklearn.linear_model import MultiTaskElasticNet #Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer
from sklearn.linear_model import MultiTaskLassoCV #Multi-task L1/L2 Lasso with built-in cross-validation.
from sklearn.linear_model import MultiTaskElasticNetCV #Multi-task L1/L2 ElasticNet with built-in cross-validation.
from sklearn.linear_model import OrthogonalMatchingPursuit #Orthogonal Matching Pursuit model OMP)
from sklearn.linear_model import OrthogonalMatchingPursuitCV #Cross-validated Orthogonal Matching Pursuit model OMP)
from sklearn.linear_model import PassiveAggressiveClassifier #Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveRegressor #Passive Aggressive Regressor
from sklearn.linear_model import Perceptron #Read more in the User Guide.
from sklearn.linear_model import RandomizedLasso #Randomized Lasso.
from sklearn.linear_model import RandomizedLogisticRegression #Randomized Logistic Regression
from sklearn.linear_model import RANSACRegressor #RANSAC RANdom SAmple Consensus) algorithm.
from sklearn.linear_model import Ridge #Linear least squares with l2 regularization.
from sklearn.linear_model import RidgeClassifier #Classifier using Ridge regression.
from sklearn.linear_model import RidgeClassifierCV #Ridge classifier with built-in cross-validation.
from sklearn.linear_model import RidgeCV #Ridge regression with built-in cross-validation.
from sklearn.linear_model import SGDClassifier #Linear classifiers SVM, logistic regression, a from o from ) with SGD training.
from sklearn.linear_model import SGDRegressor #Linear model fitted by minimizing a regularized empirical loss with SGD
from sklearn.linear_model import TheilSenRegressor #Theil-Sen Estimator: robust multivariate regression model.
from sklearn.linear_model import lars_path #Compute Least Angle Regression or Lasso path using LARS algorithm [1]
from sklearn.linear_model import lasso_path #Compute Lasso path with coordinate descent
from sklearn.linear_model import lasso_stability_path #Stabiliy path based on randomized Lasso estimates
from sklearn.linear_model import orthogonal_mp #Orthogonal Matching Pursuit OMP)
from sklearn.linear_model import orthogonal_mp_gram #Gram Orthogonal Matching Pursuit OMP)

#Manifold Learning
from sklearn.manifold import LocallyLinearEmbedding #Locally Linear Embedding
from sklearn.manifold import Isomap #Isomap Embedding
from sklearn.manifold import MDS #Multidimensional scaling
from sklearn.manifold import SpectralEmbedding #Spectral embedding for non-linear dimensionality reduction.
from sklearn.manifold import TSNE #t-distributed Stochastic Neighbor Embedding.
from sklearn.manifold import locally_linear_embedding #Perform a Locally Linear Embedding analysis on the data.
from sklearn.manifold import spectral_embedding #Project the sample on the first eigenvectors of the graph Laplacian.

#Metrics
#Classification metrics
from sklearn.metrics import accuracy_score #Accuracy classification score.
from sklearn.metrics import auc #Compute Area Under the Curve AUC) using the trapezoidal rule
from sklearn.metrics import average_precision_score #Compute average precision AP) from prediction scores
from sklearn.metrics import brier_score_loss #Compute the Brier score.
from sklearn.metrics import classification_report #Build a text report showing the main classification metrics
from sklearn.metrics import confusion_matrix #Compute confusion matrix to evaluate the accuracy of a classification
from sklearn.metrics import f1_score #Compute the F1 score, also known as balanced F-score or F-measure
from sklearn.metrics import fbeta_score #Compute the F-beta score
from sklearn.metrics import hamming_loss #Compute the average Hamming loss.
from sklearn.metrics import hinge_loss #Average hinge loss non-regularized)
from sklearn.metrics import jaccard_similarity_score #Jaccard similarity coefficient score
from sklearn.metrics import log_loss #Log loss, aka logistic loss or cross-entropy loss.
from sklearn.metrics import matthews_corrcoef #Compute the Matthews correlation coefficient MCC) for binary classes
from sklearn.metrics import precision_recall_curve #Compute precision-recall pairs for different probability thresholds
from sklearn.metrics import precision_recall_fscore_support #Compute precision, recall, F-measure and support for each class
from sklearn.metrics import precision_score #Compute the precision
from sklearn.metrics import recall_score #Compute the recall
from sklearn.metrics import roc_auc_score #Compute Area Under the Curve AUC) from prediction scores
from sklearn.metrics import roc_curve #Compute Receiver operating characteristic ROC)
from sklearn.metrics import zero_one_loss #Zero-one classification loss.
from sklearn.metrics import brier_score_loss #Compute the Brier score.
#Regression metrics
from sklearn.metrics import explained_variance_score #Explained variance regression score function
from sklearn.metrics import mean_absolute_error #Mean absolute error regression loss
from sklearn.metrics import mean_squared_error #Mean squared error regression loss
from sklearn.metrics import median_absolute_error #Median absolute error regression loss
from sklearn.metrics import r2_score #R^2 coefficient of determination) regression score function.
#Clustering metrics
from sklearn.metrics import adjusted_mutual_info_score #Adjusted Mutual Information between two clusterings
from sklearn.metrics import adjusted_rand_score #Rand index adjusted for chance
from sklearn.metrics import completeness_score #Completeness metric of a cluster labeling given a ground truth
from sklearn.metrics import homogeneity_completeness_v_measure #Compute the homogeneity and completeness and V-Measure scores at once
from sklearn.metrics import homogeneity_score #Homogeneity metric of a cluster labeling given a ground truth
from sklearn.metrics import mutual_info_score #Mutual Information between two clusterings
from sklearn.metrics import normalized_mutual_info_score #Normalized Mutual Information between two clusterings
from sklearn.metrics import silhouette_score #Compute the mean Silhouette Coefficient of all samples.
from sklearn.metrics import silhouette_samples #Compute the Silhouette Coefficient for each sample.
from sklearn.metrics import v_measure_score #V-measure cluster labeling given a ground truth.
#Pairwise metrics
from sklearn.metrics.pairwise import additive_chi2_kernel #Computes the additive chi-squared kernel between observations in X and Y
from sklearn.metrics.pairwise import chi2_kernel #Computes the exponential chi-squared kernel X and Y.
from sklearn.metrics.pairwise import distance_metrics #Valid metrics for pairwise_distances.
from sklearn.metrics.pairwise import euclidean_distances #Considering the rows of X and Y=X) as vectors, compute the distance matrix between each pair of vectors.
from sklearn.metrics.pairwise import kernel_metrics #Valid metrics for pairwise_kernels
from sklearn.metrics.pairwise import linear_kernel #Compute the linear kernel between X and Y.
from sklearn.metrics.pairwise import manhattan_distances #Compute the L1 distances between the vectors in X and Y.
from sklearn.metrics.pairwise import pairwise_distances #Compute the distance matrix from a vector array X and optional Y.
from sklearn.metrics.pairwise import pairwise_kernels #Compute the kernel between arrays X and optional array Y.
from sklearn.metrics.pairwise import polynomial_kernel #Compute the polynomial kernel between X and Y:
from sklearn.metrics.pairwise import rbf_kernel #Compute the rbf gaussian) kernel between X and Y:
from sklearn.metrics.pairwise import laplacian_kernel #Compute the laplacian kernel between X and Y.
from sklearn.metrics import pairwise_distances #Compute the distance matrix from a vector array X and optional Y.
from sklearn.metrics import pairwise_distances_argmin #Compute minimum distances between one point and a set of points.
from sklearn.metrics import pairwise_distances_argmin_min #Compute minimum distances between one point and a set of points.

#Gaussian Mixture Models
from sklearn.mixture import GMM #Gaussian Mixture Model
from sklearn.mixture import DPGMM #variational Inference for the Infinite Gaussian Mixture Model.
from sklearn.mixture import VBGMM #Variational Inference for the Gaussian Mixture Model

#Naive Bayes
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayes (GaussianNB)
from sklearn.naive_bayes import MultinomialNB #Naive Bayes classifier for multinomial models
from sklearn.naive_bayes import BernoulliNB #Naive Bayes classifier for multivariate Bernoulli models.

#Nearest Neighbors
from sklearn.neighbors import NearestNeighbors #Unsupervised learner for implementing neighbor searches.
from sklearn.neighbors import KNeighborsClassifier #Classifier implementing the k-nearest neighbors vote.
from sklearn.neighbors import RadiusNeighborsClassifier #Classifier implementing a vote among neighbors within a given radius
from sklearn.neighbors import KNeighborsRegressor #Regression based on k-nearest neighbors.
from sklearn.neighbors import RadiusNeighborsRegressor #Regression based on neighbors within a fixed radius.
from sklearn.neighbors import NearestCentroid #Nearest centroid classifier.
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
from sklearn.neighbors import LSHForest #Performs approximate nearest neighbor search using LSH forest.
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KernelDensity #Kernel Density Estimation
from sklearn.neighbors import kneighbors_graph #Computes the weighted) graph of k-Neighbors for points in X
from sklearn.neighbors import radius_neighbors_graph #Computes the weighted) graph of Neighbors for points in X

##Neural network models
#from sklearn_theano.feature_extraction import GoogLeNetClassifier
#from sklearn_theano.feature_extraction import OverfeatClassifier

#Cross decomposition
from sklearn.cross_decomposition import PLSRegression #PLS regression
from sklearn.cross_decomposition import PLSCanonical #PLSCanonical implements the 2 blocks canonical PLS of the original Wold algorithm [Tenenhaus 1998] p from 204, referred as PLS-C2A in [Wegelin 2000].
from sklearn.cross_decomposition import CCA #CCA Canonical Correlation Analysis.
from sklearn.cross_decomposition import PLSSVD #Partial Least Square SVD

#Pipeline
from sklearn.pipeline import Pipeline #Pipeline of transforms with a final estimator.
from sklearn.pipeline import FeatureUnion #Concatenates results of multiple transformer objects.

#Preprocessing and Normalization
from sklearn.preprocessing import Binarizer #Binarize data (set feature values to 0 or 1) according to a threshold
from sklearn.preprocessing import FunctionTransformer #Constructs a transformer from an arbitrary callable.
from sklearn.preprocessing import Imputer #Imputation transformer for completing missing values.
from sklearn.preprocessing import KernelCenterer #Center a kernel matrix
from sklearn.preprocessing import LabelBinarizer #Binarize labels in a one-vs-all fashion
from sklearn.preprocessing import LabelEncoder #Encode labels with value between 0 and n_classes-1.
from sklearn.preprocessing import MultiLabelBinarizer #Transform between iterable of iterables and a multilabel format
from sklearn.preprocessing import MaxAbsScaler #Scale each feature by its maximum absolute value.
from sklearn.preprocessing import MinMaxScaler #Transforms features by scaling each feature to a given range.
from sklearn.preprocessing import Normalizer #Normalize samples individually to unit norm.
from sklearn.preprocessing import OneHotEncoder #Encode categorical integer features using a one-hot aka one-of-K scheme.
from sklearn.preprocessing import PolynomialFeatures #Generate polynomial and interaction features.
from sklearn.preprocessing import RobustScaler #Scale features using statistics that are robust to outliers.
 # NOTES: with_mean=True,with_std=True equivalent to centering and uv scaling
from sklearn.preprocessing import StandardScaler #Standardize features by removing the mean and scaling to unit variance

#Support Vector Machines
from sklearn.svm import SVC #C-Support Vector Classification.
from sklearn.svm import LinearSVC #Linear Support Vector Classification.
from sklearn.svm import NuSVC #Nu-Support Vector Classification.
from sklearn.svm import SVR #Epsilon-Support Vector Regression.
from sklearn.svm import LinearSVR #Linear Support Vector Regression.
from sklearn.svm import NuSVR #Nu Support Vector Regression.
from sklearn.svm import OneClassSVM #Unsupervised Outlier Detection.
from sklearn.svm import l1_min_c #Return the lowest bound for C such that for C in l1_min_C, infinity) the model is guaranteed not to be empty.

#Decision Trees
from sklearn.tree import DecisionTreeClassifier #A decision tree classifier.
from sklearn.tree import DecisionTreeRegressor #A decision tree regressor.
from sklearn.tree import ExtraTreeClassifier #An extremely randomized tree classifier.
from sklearn.tree import ExtraTreeRegressor #An extremely randomized tree regressor.
from sklearn.tree import export_graphviz #Export a decision tree in DOT format from 
