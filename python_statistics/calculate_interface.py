# calculate classes to include
from .calculate_smoothingFunctions import calculate_smoothingFunctions
from .calculate_statisticsSampledPoints import calculate_statisticsSampledPoints
from .calculate_statisticsDescriptive import calculate_statisticsDescriptive
from .calculate_biomass import calculate_biomass
from .calculate_curveFitting import calculate_curveFitting
from .calculate_histogram import calculate_histogram
from .calculate_clustering import calculate_clustering
from .calculate_correlation import calculate_correlation
from .calculate_count import calculate_count
from .calculate_classification import calculate_classification



class calculate_interface(
        calculate_statisticsDescriptive,
        calculate_smoothingFunctions,
        calculate_statisticsSampledPoints,
        calculate_biomass,
        calculate_clustering,
        calculate_curveFitting,
        calculate_histogram):
    '''conveniency class that wraps all of the calculate methods using various python open-source statistics modules
    '''
    pass;