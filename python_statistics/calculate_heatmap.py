from .calculate_dependencies import *
from .calculate_base import calculate_base

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from listDict.listDict import listDict

class calculate_heatmap(calculate_base):
    def __init__(self,
                 listDict_I=None,
                 data_I = None,
                 heatmap_I=[],dendrogram_col_I=[],dendrogram_row_I=[]):
        '''
        INPUT:
        heatmap_I = heatmap data
        dendrogram_I = dendrogram data
        '''
        if listDict_I: self.listDict=listDict_I;
        else: self.listDict = None;
        if data_I: self.data=data_I;
        else: self.data = {};
        if heatmap_I:
            self.heatmap=heatmap_I;
        else:
            self.heatmap = [];
        if dendrogram_col_I:
            self.dendrogram_col=dendrogram_col_I;
        else:
            self.dendrogram_col = [];
        if dendrogram_row_I:
            self.dendrogram_row=dendrogram_row_I;
        else:
            self.dendrogram_row = [];

    def clear_data(self):
        self.data = {};
        del self.heatmap[:];
        del self.dendrogram_col[:];
        del self.dendrogram_row[:];

    # heatmap
    def _make_heatmap(self,data_I,row_labels_I,column_labels_I,
                row_pdist_metric_I='euclidean',row_linkage_method_I='complete',
                col_pdist_metric_I='euclidean',col_linkage_method_I='complete',
                ):
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
        #Y = linkage(D1, method=row_linkage_method_I, metric=row_pdist_metric_I)
        Z1 = dendrogram(Y, labels=dm.index)

        Y = linkage(D2, method=col_linkage_method_I)
        #Y = linkage(D2, method=col_linkage_method_I, metric=col_pdist_metric_I)
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

    def make_heatmap(self,
                data_I,row_label_I,column_label_I,value_label_I,
                row_pdist_metric_I='euclidean',row_linkage_method_I='complete',
                col_pdist_metric_I='euclidean',col_linkage_method_I='complete',
                filter_rows_I=[],
                filter_columns_I=[],
                order_rowsFromTemplate_I=[],
                order_columnsFromTemplate_I=[],):
        '''Make the heatmap
        INPUT: 
        data_I = [{}]
        row_label_I = column_id of the row labels
        column_label_I = column_id of the column labels
        value_label_I = column_id of the value label
        OUTPUT:
        data_O = list of values ordered according to (len(row_label_unique),len(column_label_unique))
        row_labels_O = row labels of data_O
        column_labels_O = column labels of data_O
        '''
        
        #make the listdict
        data_listDict = listDict(data_I);
        data_listDict.convert_listDict2DataFrame();
        value_label = value_label_I;
        row_labels = [row_label_I];
        column_labels = [column_label_I];

        #filter in rows/columns
        if filter_rows_I:
            data_listDict.filterIn_byDictList({row_label_I:filter_rows_I,
                                           });
        if filter_columns_I:
            data_listDict.filterIn_byDictList({column_label_I:filter_columns_I,
                                           });
        
        #make the pivot table
        data_listDict.set_pivotTable(
            value_label_I=value_label,
            row_labels_I=row_labels,
            column_labels_I=column_labels
            );

        #sort rows/columns
        if order_rowsFromTemplate_I:
            data_listDict.order_indexFromTemplate_pivotTable(template_I=order_rowsFromTemplate_I,axis_I=0);
        if order_columnsFromTemplate_I:
            data_listDict.order_indexFromTemplate_pivotTable(template_I=order_columnsFromTemplate_I,axis_I=1);
        
        #make the heatmap data matrix
        data_O = data_listDict.get_dataMatrix();
        row_labels_unique = data_listDict.get_rowLabels_asArray();
        column_labels_unique = data_listDict.get_columnLabels_asArray();

        #make the heatmap
        heatmap_O,dendrogram_col_O,dendrogram_row_O = self._make_heatmap(data_O,row_labels_unique,column_labels_unique,
            row_pdist_metric_I=row_pdist_metric_I,row_linkage_method_I=row_linkage_method_I,
            col_pdist_metric_I=col_pdist_metric_I,col_linkage_method_I=col_linkage_method_I);
        return heatmap_O,dendrogram_col_O,dendrogram_row_O;

    def convert_idCoord2NodeDistance_dendrogramCol(self):
        '''
        '''

        icoord,dcoord,colors,leaves,ivl=self.dendrogram_col['icoord'],\
            self.dendrogram_col['dcoord'],\
            self.dendrogram_col['colors'],\
            self.dendrogram_col['leaves'],\
            self.dendrogram_col['ivl'];
        nodes_O = self.convert_idCoord2NodeDistance(
            icoord,dcoord,colors,leaves,ivl
            );
        return nodes_O;

    def convert_idCoord2NodeDistance_dendrogramRow(self):
        '''
        '''

        icoord,dcoord,colors,leaves,ivl=self.dendrogram_row['icoord'],\
            self.dendrogram_row['dcoord'],\
            self.dendrogram_row['colors'],\
            self.dendrogram_row['leaves'],\
            self.dendrogram_row['ivl'];
        nodes_O = self.convert_idCoord2NodeDistance(
            icoord,dcoord,colors,leaves,ivl
            );
        return nodes_O;

    def convert_idCoord2NodeDistance(self,
        icoord_I,dcoord_I,colors_I,leaves_I,ivl_I,
        convert_colors_hex_I = True
        ):
        '''convert i/dcoord to nodes and distances
        '''
        #matplotlib colors
        #b: blue
        #g: green
        #r: red
        #c: cyan
        #m: magenta
        #y: yellow
        #k: black
        #w: white
        colors_hex_conv = {'b':'#0000CC',
                           'r':'#CC0000',
                           'g':'#009900',
                           'm':'#6600CC',
                           'c':'#00CCCC',
                           'y':'#FFFF00',
                           'k':'#000000',
                           'w':'FFFFFF',};

        #dcoord_all = [];
        #for row in dcoord_I:
        #    for d in row:
        #        if d != 0:
        #            dcoord_all.append(d);
        #lowest_distance_value = np.min(dcoord_all)*0.5;
        ##lowest_distance_value = 1e-3;

        nodes_O = [];
        leaf_cnt = 0;
        parents = [];
        names = [];
        for icoord_cnt, icoord in enumerate(icoord_I):
            # calculate the distance
            distance1 = np.abs(dcoord_I[icoord_cnt][1]-dcoord_I[icoord_cnt][0]);
            distance2 = np.abs(dcoord_I[icoord_cnt][3]-dcoord_I[icoord_cnt][2]);
            distances = [distance1,distance2];
            if sum(distances)==0:
                #check for [0,0,0,0] dcoord
                #insufficient precision for recording low value dcoord's
                continue;
                #distances = [lowest_distance_value,lowest_distance_value];
            # store the unique icoord
            icoord1 = icoord[0];
            icoord2 = icoord[2];
            assert(icoord1!=icoord2); # check that the order is valid
            icoords = [icoord1,icoord2];
            dcoords = [[dcoord_I[icoord_cnt][1],dcoord_I[icoord_cnt][0]],[dcoord_I[icoord_cnt][3],dcoord_I[icoord_cnt][2]]]
            # record the nodes
            for link in range(len(icoords)):
                tmp={};
                tmp['idcoord_index'] = icoord_cnt;
                tmp['length'] = distances[link];
                if convert_colors_hex_I: tmp['color'] = colors_hex_conv[colors_I[icoord_cnt]]
                else: tmp['color'] = colors_I[icoord_cnt];
                p1_index = dcoord_I[icoord_cnt].index(max(dcoord_I[icoord_cnt]))

                #p1_str = str(icoord[p1_index]);
                #p2_str = str(max(dcoord_I[icoord_cnt]));
                #tmp['parent'] = '%s-%s'%(p1_str,p2_str);

                tmp['parent'] = str(np.round(np.mean(icoords),3));
                # check for an original observation
                if 0 in dcoords[link]:
                    if len(ivl_I)<=leaf_cnt:
                        print("check leaf # " + str(leaf_cnt));
                        tmp['name'] = str(np.round(icoords[link],3));
                        leaf_cnt+=1;
                    else:
                        tmp['name'] = ivl_I[leaf_cnt];
                        leaf_cnt+=1;
                else:
                    tmp['name'] = str(np.round(icoords[link],3));
                    #tmp['name'] = '%s-%s'%(p1_str,p2_str);
                nodes_O.append(tmp);
                parents.append(tmp['parent'])
                names.append(tmp['name'])
            ##debugging:
            #print(leaves_I[icoord_cnt],ivl_I[icoord_cnt],icoords)
            #for xs, ys, color in zip(icoord_I[:icoord_cnt], dcoord_I[:icoord_cnt], colors_I[:icoord_cnt]):
            #    plt.plot(xs, ys,  color)
            #plt.show()
        #add in dummy node
        tmp={};
        tmp['idcoord_index'] = icoord_cnt;
        tmp['length'] = distances[link];
        if convert_colors_hex_I: tmp['color'] = colors_hex_conv[colors_I[icoord_cnt]]
        else: tmp['color'] = colors_I[icoord_cnt];
        tmp['name']=str(np.round(np.mean(icoords),3));
        tmp['parent']='';
        nodes_O.append(tmp);
        assert(len(ivl_I)==leaf_cnt)
        return nodes_O;

    def plot_dendrogram(icoord_I, dcoord_I, colors_I):
        '''plot a dendrogram using matplot lib'''
        for xs, ys, color in zip(icoord_I, dcoord_I, colors_I):
            plt.plot(xs, ys,  color)
        plt.show();