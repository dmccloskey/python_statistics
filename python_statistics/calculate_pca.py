from .calculate_dependencies import *
from .calculate_base import calculate_base
class calculate_pca(calculate_base):
    def calculate_pca(self):
        '''calculate PCA
        sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)

        from sklearn.datasets import load_iris

        iris = load_iris()
        X, y = iris.data, iris.target

        from sklearn.decomposition import PCA
        pca = PCA()

        pca.fit(X_blob)

        X_pca = pca.transform(X_blob)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, linewidths=0, s=30)
        plt.xlabel("first principal component")
        plt.ylabel("second principal component")

        '''
        pass;

    def extract_scoresAndLoadings_2D(self,data_scores,data_loadings,PCs):
        '''Extract out the scores and loadings
        INPUT:
        data_scores = listDict of pca/pls scores
        data_loadings = listDict of pca/pls loadings
        PCs = [[],[],...] of integers, describing the 2D PC plots
            E.G. PCs = [[1,2],[1,3],[2,3]];
        OUTPUT:
        data_scores_O = {'[1,2]':[],'[1,3]':[],'[2,3]':[],...} where each [] is a listDict of the data from PCs e.g. 1,2
        data_loadings_O = {'[1,2]':[],'[1,3]':[],'[2,3]':[],...}
        '''
        data_scores_O,data_loadings_O = {},{};
        for PC_cnt,PC in enumerate(PCs):
            #extract out the scores
            data_scores_O[str(PC)]=[];
            for cnt,d in enumerate(data_scores[PC[0]][:]):
                if d['sample_name_short'] != data_scores[PC[1]][cnt]['sample_name_short'] and d['calculated_concentration_units'] != data_scores[PC[1]][cnt]['calculated_concentration_units']:
                    print('data is not in the correct order');
                tmp = copy.copy(d);
                tmp['score_' + str(PC[0])] = d['score'];
                tmp['var_proportion_'+str(PC[0])] = d['var_proportion'];
                tmp['var_cumulative_'+str(PC[0])] = d['var_cumulative'];
                tmp['axislabel'+str(PC[0])] = 'PC' + str(PC[0]) + ' [' + str(round(d['var_proportion']*100,2)) + '%]';
                tmp['score_' + str(PC[1])] = data_scores[PC[1]][cnt]['score'];
                tmp['var_proportion_'+str(PC[1])] = data_scores[PC[1]][cnt]['var_proportion'];
                tmp['var_cumulative_'+str(PC[1])] = data_scores[PC[1]][cnt]['var_cumulative'];
                tmp['axislabel'+str(PC[1])] = 'PC' + str(PC[1]) + ' [' + str(round(data_scores[PC[1]][cnt]['var_proportion']*100,2)) + '%]';
                del tmp['score'];
                del tmp['axis'];
                del tmp['var_proportion'];
                del tmp['var_cumulative'];
                data_scores_O[str(PC)].append(tmp);
            #extract out the loadings
            data_loadings_O[str(PC)]=[];
            for cnt,d in enumerate(data_loadings[PC[0]][:]):
                if d['component_name'] != data_loadings[PC[1]][cnt]['component_name'] and d['calculated_concentration_units'] != data_loadings[PC[1]][cnt]['calculated_concentration_units']:
                    print('data is not in the correct order');
                tmp = copy.copy(d);
                tmp['loadings_' + str(PC[0])] = d['loadings'];
                tmp['axislabel'+str(PC[0])] = 'Loadings' + str(PC[0]);
                tmp['loadings_' + str(PC[1])] = data_loadings[PC[1]][cnt]['loadings'];
                tmp['axislabel'+str(PC[1])] = 'Loadings' + str(PC[1]);
                del tmp['loadings'];
                del tmp['axis'];
                data_loadings_O[str(PC)].append(tmp);
        return data_scores_O,data_loadings_O;

