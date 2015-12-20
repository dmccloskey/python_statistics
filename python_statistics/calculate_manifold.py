from .calculate_dependencies import *
from .calculate_base import calculate_base
class calculate_manifold(calculate_base):
    def calculate_manifold(self,data_I,method_I):
        '''Calculate the dimensionality reduction using manifold methods
        Methods:
        from sklearn.manifold import isomap, TSNE, ...
        '''


