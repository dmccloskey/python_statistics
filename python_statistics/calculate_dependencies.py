from scipy.stats import linregress
import scipy.stats
from scipy.sparse.linalg import svds
from math import ceil, sqrt

import matplotlib.pyplot as plt
from scipy import linspace, sin
from scipy.interpolate import splrep, splev
import numpy as np
import numpy.random as npr
import json
import copy

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

import pandas as pd

from .cookb_signalsmooth import smooth
from .legendre_smooth import legendre_smooth
from Bio.Statistics import lowess