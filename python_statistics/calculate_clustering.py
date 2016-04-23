from .calculate_dependencies import *
from .calculate_base import calculate_base

from time import time
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, KMeans

import scipy.spatial.distance

class calculate_clustering(calculate_base):
    # heatmap
    def calculate_heatmap(self,data_I,row_labels_I,column_labels_I,
                row_pdist_metric_I='euclidean',row_linkage_method_I='complete',
                col_pdist_metric_I='euclidean',col_linkage_method_I='complete'):
        '''Generate a heatmap using pandas and scipy'''

        """dendrogram documentation:
        linkage Methods:
        single(y)	Performs single/min/nearest linkage on the condensed distance matrix y
        complete(y)	Performs complete/max/farthest point linkage on a condensed distance matrix
        average(y)	Performs average/UPGMA linkage on a condensed distance matrix
        weighted(y)	Performs weighted/WPGMA linkage on the condensed distance matrix.
        centroid(y)	Performs centroid/UPGMC linkage.
        median(y)	Performs median/WPGMC linkage.
        ward(y)	Performs Ward's linkage on a condensed or redundant distance matrix.
        Output:
        'color_list': A list of color names. The k?th element represents the color of the k?th link.
        'icoord' and 'dcoord':  Each of them is a list of lists. Let icoord = [I1, I2, ..., Ip] where Ik = [xk1, xk2, xk3, xk4] and dcoord = [D1, D2, ..., Dp] where Dk = [yk1, yk2, yk3, yk4], then the k?th link painted is (xk1, yk1) - (xk2, yk2) - (xk3, yk3) - (xk4, yk4).
        'ivl':  A list of labels corresponding to the leaf nodes.
        'leaves': For each i, H[i] == j, cluster node j appears in position i in the left-to-right traversal of the leaves, where \(j < 2n-1\) and \(i < n\). If j is less than n, the i-th leaf node corresponds to an original observation. Otherwise, it corresponds to a non-singleton cluster."""

        #parse input into col_labels and row_labels
        #TODO: pandas is not needed for this.
        mets_data = pd.DataFrame(data=data_I, index=row_labels_I, columns=column_labels_I)

        mets_data = mets_data.dropna(how='all').fillna(0.)
        #mets_data = mets_data.replace([np.inf], 10.)
        #mets_data = mets_data.replace([-np.inf], -10.)
        col_labels = list(mets_data.columns)
        row_labels = list(mets_data.index)

        #perform the custering on the both the rows and columns
        dm = mets_data
        D1 = squareform(pdist(dm, metric=row_pdist_metric_I))
        D2 = squareform(pdist(dm.T, metric=col_pdist_metric_I))

        Y = linkage(D1, method=row_linkage_method_I)
        Z1 = dendrogram(Y, labels=dm.index)

        Y = linkage(D2, method=col_linkage_method_I)
        Z2 = dendrogram(Y, labels=dm.columns)

        #parse the output
        col_leaves = Z2['leaves'] # no hclustering; same as heatmap_data['col']
        row_leaves = Z1['leaves'] # no hclustering; same as heatmap_data['row']
        col_colors = Z2['color_list'] # no hclustering; same as heatmap_data['col']
        row_colors = Z1['color_list'] # no hclustering; same as heatmap_data['row']
        col_icoord = Z2['icoord'] # no hclustering; same as heatmap_data['col']
        row_icoord = Z1['icoord'] # no hclustering; same as heatmap_data['row']
        col_dcoord = Z2['dcoord'] # no hclustering; same as heatmap_data['col']
        row_dcoord = Z1['dcoord'] # no hclustering; same as heatmap_data['row']
        col_ivl = Z2['ivl'] # no hclustering; same as heatmap_data['col']
        row_ivl = Z1['ivl'] # no hclustering; same as heatmap_data['row']

        #heatmap data matrix
        heatmap_data_O = []
        for i,r in enumerate(mets_data.index):
            for j,c in enumerate(mets_data.columns):
                #heatmap_data.append({"col": j+1, "row": i+1, "value": mets_data.ix[r][c]})
                tmp = {"col_index": j, "row_index": i, "value": mets_data.ix[r][c],
                       'row_label': r,'col_label':c,
                       'col_leaves':col_leaves[j],
                        'row_leaves':row_leaves[i],
                        'col_pdist_metric':col_pdist_metric_I,
                        'row_pdist_metric':row_pdist_metric_I,
                        'col_linkage_method':col_linkage_method_I,
                        'row_linkage_method':row_linkage_method_I,
                       };
                heatmap_data_O.append(tmp)

        dendrogram_col_O = {'leaves':col_leaves,
                        'icoord':col_icoord,
                        'dcoord':col_dcoord,
                        'ivl':col_ivl,
                        'colors':col_colors,
                        'pdist_metric':col_pdist_metric_I,
                        'linkage_method':col_linkage_method_I};

        dendrogram_row_O = {
                        'leaves':row_leaves,
                        'icoord':row_icoord,
                        'dcoord':row_dcoord,
                        'ivl':row_ivl,
                        'colors':row_colors,
                        'pdist_metric':row_pdist_metric_I,
                        'linkage_method':row_linkage_method_I};
        return heatmap_data_O,dendrogram_col_O,dendrogram_row_O;
    
    def calculate_clusters(self,data_I,cluster_method_I):
        '''call a clustering algorithm on the data
        Cluster methods available:
        sklearn.cluster.KMeans: 
        The simplest, yet effective clustering algorithm. Needs to be provided with the number of clusters in advance, and assumes that the data is normalized as input (but use a PCA model as preprocessor).
        sklearn.cluster.MeanShift: 
        Can find better looking clusters than KMeans but is not scalable to high number of samples.
        sklearn.cluster.DBSCAN: 
        Can detect irregularly shaped clusters based on density, i.e. sparse regions in the input space are likely to become inter-cluster boundaries. Can also detect outliers (samples that are not part of a cluster).
        sklearn.cluster.AffinityPropagation: 
        Clustering algorithm based on message passing between data points.
        sklearn.cluster.SpectralClustering: 
        KMeans applied to a projection of the normalized graph Laplacian: finds normalized graph cuts if the affinity matrix is interpreted as an adjacency matrix of a graph.
        sklearn.cluster.Ward: 
        Ward implements hierarchical clustering based on the Ward algorithm, a variance-minimizing approach. At each step, it minimizes the sum of squared differences within all clusters (inertia criterion).
        '''

        # Generate sample data
        np.random.seed(0);
        pass;

    #k-means
    def calculate_KMeans(self,data_I,n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1):
        '''calculate the kmeans clusters
        sklearn.cluster.k_means(X, n_clusters, init='k-means++', precompute_distances='auto', n_init=10, max_iter=300, verbose=False, tol=0.0001, random_state=None, copy_x=True, n_jobs=1, return_n_iter=False)
        
        E.g.
        from sklearn.datasets import make_blobs
        X, y = make_blobs(random_state=42)
        X.shape

        plt.scatter(X[:, 0], X[:, 1])

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)

        plt.scatter(X[:, 0], X[:, 1], c=labels)

        E.G. http://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#example-cluster-plot-mini-batch-kmeans-py
        '''
        pass
    def calculate_MiniBatchKMeans(self,data_I,
                n_clusters=8, init='k-means++',
                max_iter=100, batch_size=100,
                verbose=0, compute_labels=True,
                random_state=None, tol=0.0,
                max_no_improvement=10, init_size=None,
                n_init=3, reassignment_ratio=0.01
                ):
        '''calculate MiniBatchKMeans clusters
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans.fit
        EXAMPLE:
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
        t0 = time.time()
        mbk.fit(X) where X : array-like, shape = [n_samples, n_features]
        t_mini_batch = time.time() - t0
        mbk_means_labels = mbk.labels_
        mbk_means_cluster_centers = mbk.cluster_centers_
        mbk_means_labels_unique = np.unique(mbk_means_labels)

        ax = fig.add_subplot(1, 3, 2)
        for k, col in zip(range(n_clusters), colors):
            my_members = mbk_means_labels == order[k]
            cluster_center = mbk_means_cluster_centers[order[k]]
            ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)
        ax.set_title('MiniBatchKMeans')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
                 (t_mini_batch, mbk.inertia_))

        http://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#example-cluster-plot-mini-batch-kmeans-py
        '''

        mbk = MiniBatchKMeans(
                n_clusters=n_clusters, init=init,
                max_iter=max_iter, batch_size=batch_size,
                verbose=verbose, compute_labels=compute_labels,
                random_state=random_state, tol=tol,
                max_no_improvement=max_no_improvement, init_size=init_size,
                n_init=n_init, reassignment_ratio=reassignment_ratio)
        t0 = time.time()
        mbk.fit(data_I)
        t_mini_batch = time.time() - t0
        mbk_means_labels = mbk.labels_
        mbk_means_cluster_centers = mbk.cluster_centers_
        mbk_means_labels_unique = np.unique(mbk_means_labels);
        return mbk_means_labels,mbk_means_cluster_centers,mbk_means_labels_unique;

    def score_clusters(self,data_I,labels_I,metric_I):
        '''Score the clustering algorithm
        Methods:
        from sklearn.metrics import confusion_matrix,
            accuracy_score
            adjusted_rand_score
        
        '''

    def benchmark_dataModel(self,
                data_model_I=None,raise_I=False,
                metrics_I=[],metric_parameters_I=None):
        '''
        benchmark a scikit learn data model
        http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py
        INPUT:
        metrics_I = list of metrics
        metric_parameters_I = list of metric dictionary parameters
        OUTPUT:
        EXAMPLE:
        bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)
              '''
        metrics_O = {};

        if data_model_I: data_model=data_model_I;
        else: data_model = self.data_model;
        data_model = self.get_finalEstimator(data_model);

        y_predicted=None;
        try:
            if hasattr(data_model, 'labels_'):
                y_predicted = data_model.labels_.astype(np.int)
            else:
                y_predicted = data_model.predict(X)
            for i,metric in enumerate(metrics_I):
                model_obj = scikitLearn_obj.get_scikitLearnObjectFromStr2scikitLearnObjectDict(metric);
                metric_O = model_object(self.data_train['response'], y_predicted,**metric_parameters_I[i])
                metrics_O[metric]=metric_O;
            return metrics_O;
        except Exception as e:
            if raise_I: raise;
            else: print(e);

    #SOM

    #gap statistic
    # gap.py

    def calculate_gapStatistic(data, refs=None, nrefs=20, ks=range(1,11)):
        """
        Compute the Gap statistic for an nxm dataset in data.
        Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
        or state the number k of reference distributions in nrefs for automatic generation with a
        uniformed distribution within the bounding box of data.
        Give the list of k-values for which you want to compute the statistic in ks.
        """
    
        # (c) 2013 Mikael Vejdemo-Johansson
        # BSD License
        #
        # SciPy function to compute the gap statistic for evaluating k-means clustering.
        # Gap statistic defined in
        # Tibshirani, Walther, Hastie:
        #  Estimating the number of clusters in a data set via the gap statistic
        #  J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423

        import scipy
        #import scipy.cluster.vq
        #TODO: change to KMeans (scikit-learn)
        shape = data.shape
        if refs==None:
            tops = data.max(axis=0)
            bots = data.min(axis=0)
            dists = scipy.matrix(scipy.diag(tops-bots))
            rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
            for i in range(nrefs):
                rands[:,:,i] = rands[:,:,i]*dists+bots
        else:
            rands = refs

        gaps = scipy.zeros((len(ks),))
        for (i,k) in enumerate(ks):
            #(kmc,kml) = scipy.cluster.vq.kmeans2(data, k)
            #TODO: change to KMeans (scikit-learn)
            disp = sum([scipy.spatial.distance.euclidean(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])

            refdisps = scipy.zeros((rands.shape[2],))
            for j in range(rands.shape[2]):
                #(kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k)
                #TODO: change to KMeans (scikit-learn)
                refdisps[j] = sum([scipy.spatial.distance.euclidean(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
            gaps[i] = scipy.log(scipy.mean(refdisps))-scipy.log(disp)
           #gaps[i] = scipy.mean(scipy.log(refdisps))-scipy.log(disp) #ERROR?
        return gaps;