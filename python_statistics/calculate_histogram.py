from .calculate_dependencies import *
from .calculate_base import calculate_base

class calculate_histogram(calculate_base):

    # histogram and kde
    def histogram(self, data_I,
            n_bins_I=50,
            calc_bins_I=True,
            n_range_I=None,
            density_I=False):
        '''generate lower bound of the bins, the bin widths, and frequency of the data
        INPUT:
        data_I = [float,float,...] of data
        n_bins_I = float, # of bins
        calc_bins_I = boolean, if True, bins will be calculated
        n_range_I = (float,float), describing the (min,max)
        density_I = boolean, if True, the probability density will be returned
                             if False, the number of samples in each bin will be returned
        OUTPUT:
        x_O = [float,...]; the lower bound of the bin (inclusive)
        dx_O = [float,...]; the width of the bin; x + dx is the upper bound (exclusive).
        y_O = [float,...]; the count (if density_I = False), or the probability (if density_I = True).

        '''

        x_O = []; #the lower bound of the bin (inclusive)
        dx_O = []; #the width of the bin; x + dx is the upper bound (exclusive).
        y_O = []; #the count (if frequency is true), or the probability (if frequency is false).

        if calc_bins_I:
            n_bins = ceil(sqrt(len(data_I)));
        else:
            n_bins = n_bins_I;

        hist = np.histogram(data_I,n_bins);
        y_O = hist[0];
        edges = hist[1];

        for i in range(len(edges)-1):
            x_O.append(edges[i])
            dx_O.append(edges[i+1]-edges[i])

        return x_O,dx_O,y_O
    def pdf_kde(self,data_I,min_I=None,max_I=None,points_I=1000,bandwidth_I=None):
        '''compute the pdf from the kernal density estimate'''

        if min_I and max_I:
            min_point = min_I;
            max_point = max_I;
        else:
            min_point = min(data_I);
            max_point = max(data_I);

        x_grid = np.linspace(min_point, max_point, 1000)
        try:
            kde_scipy=scipy.stats.gaussian_kde(data_I, bw_method=bandwidth_I);
        except RuntimeError as e:
            print(e)
            return [0],[0];
        pdf = kde_scipy(x_grid);

        return x_grid,pdf
