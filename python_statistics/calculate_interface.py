# calculate classes to include
from .calculate_biomass import calculate_biomass
from .calculate_clustering import calculate_clustering
from .calculate_correlation import calculate_correlation
from .calculate_count import calculate_count
from .calculate_covariance import calculate_covariance
from .calculate_curveFitting import calculate_curveFitting
#from .calculate_heatmap import calculate_heatmap (has its own __init__)
from .calculate_histogram import calculate_histogram
from .calculate_importantFeatures import calculate_importantFeatures
from .calculate_outliers import calculate_outliers
from .calculate_pca import calculate_pca
from .calculate_pls import calculate_pls
from .calculate_responseClassification import calculate_responseClassification
from .calculate_smoothingFunctions import calculate_smoothingFunctions
from .calculate_statisticsDescriptive import calculate_statisticsDescriptive
from .calculate_statisticsSampledPoints import calculate_statisticsSampledPoints
from .calculate_statisticsUnivariate import calculate_statisticsUnivariate
from .data_partitioning import data_partitioning
from .data_preProcessing import data_preProcessing
from .scikitLearn_wrapper import scikitLearn_wrapper

class calculate_interface(
        calculate_biomass,
        calculate_clustering,
        calculate_correlation,
        calculate_curveFitting,
        calculate_count,
        calculate_covariance,
        #calculate_heatmap,
        calculate_histogram,
        calculate_importantFeatures,
        calculate_outliers,
        calculate_pca,
        calculate_pls,
        calculate_responseClassification,
        calculate_smoothingFunctions,
        calculate_statisticsDescriptive,
        calculate_statisticsSampledPoints,
        calculate_statisticsUnivariate,
        data_partitioning,
        data_preProcessing,
        scikitLearn_wrapper,
        ):
    '''conveniency class that wraps all of the calculate methods using various python open-source statistics modules
    '''
    pass;