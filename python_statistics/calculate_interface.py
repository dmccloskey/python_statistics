# calculate classes to include
from .calculate_biomass import calculate_biomass
from .calculate_boostedModel import calculate_boostedModel
from .calculate_classification import calculate_classification
from .calculate_clustering import calculate_clustering
from .calculate_correlation import calculate_correlation
from .calculate_count import calculate_count
from .calculate_covariance import calculate_covariance
from .calculate_curveFitting import calculate_curveFitting
from .calculate_dataSplit import calculate_dataSplit
#from .calculate_heatmap import calculate_heatmap (has its own __init__)
from .calculate_histogram import calculate_histogram
from .calculate_importantFeatures import calculate_importantFeatures
from .calculate_manifold import calculate_manifold
from .calculate_missingValues import calculate_missingValues
from .calculate_modelValidation import calculate_modelValidation
from .calculate_outliers import calculate_outliers
from .calculate_pca import calculate_pca
from .calculate_pls import calculate_pls
from .calculate_regression import calculate_regression
from .calculate_smoothingFunctions import calculate_smoothingFunctions
from .calculate_statisticsDescriptive import calculate_statisticsDescriptive
from .calculate_statisticsSampledPoints import calculate_statisticsSampledPoints
from .calculate_statisticsUnivariate import calculate_statisticsUnivariate
from .calculate_tree import calculate_tree
from .calculate_svm import calculate_svm

class calculate_interface(
        calculate_biomass,
        calculate_boostedModel,
        calculate_classification,
        calculate_clustering,
        calculate_correlation,
        calculate_curveFitting,
        calculate_count,
        calculate_covariance,
        calculate_dataSplit,
        #calculate_heatmap,
        calculate_histogram,
        calculate_importantFeatures,
        calculate_manifold,
        calculate_missingValues,
        calculate_modelValidation,
        calculate_outliers,
        calculate_pca,
        calculate_pls,
        calculate_regression,
        calculate_smoothingFunctions,
        calculate_statisticsDescriptive,
        calculate_statisticsSampledPoints,
        calculate_statisticsUnivariate,
        calculate_tree,
        calculate_svm
        ):
    '''conveniency class that wraps all of the calculate methods using various python open-source statistics modules
    '''
    pass;