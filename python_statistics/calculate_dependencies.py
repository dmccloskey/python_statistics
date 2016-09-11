#system
from math import ceil, sqrt
import json
import copy
#scipy
from scipy.stats import linregress,hypergeom
import scipy.stats
from scipy.sparse.linalg import svds
from scipy import linspace, sin
from scipy.interpolate import splrep, splev
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
#numpy
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
#pandas
import pandas as pd
#other
from .cookb_signalsmooth import smooth
from .legendre_smooth import legendre_smooth
from Bio.Statistics import lowess