from .calculate_dependencies import *
from .calculate_base import calculate_base
class calculate_svd(calculate_base):

    def calculate_svd(self):
        '''calculate SVD'''
        pass;

    def extract_UAndVMatrices_2D(self,data_u,data_v,PCs):
        '''Extract out the scores and loadings
        INPUT:
        data_u = listDict of pca/pls scores
        data_v = listDict of pca/pls loadings
        PCs = [[],[],...] of integers, describing the 2D PC plots
            E.G. PCs = [[1,2],[1,3],[2,3]];
        OUTPUT:
        data_u_O = {'[1,2]':[],'[1,3]':[],'[2,3]':[],...} where each [] is a listDict of the data from PCs e.g. 1,2
        data_v_O = {'[1,2]':[],'[1,3]':[],'[2,3]':[],...}
        '''
        data_u_O,data_v_O = {},{};
        for PC_cnt,PC in enumerate(PCs):
            #extract out the scores
            data_u_O[str(PC)]=[];
            for cnt,d in enumerate(data_u[PC[0]][:]):
                if d['sample_name_short'] != data_u[PC[1]][cnt]['sample_name_short'] and d['calculated_concentration_units'] != data_u[PC[1]][cnt]['calculated_concentration_units']:
                    print('data is not in the correct order');
                tmp = copy.copy(d);
                tmp['score_' + str(PC[0])] = d['u_matrix'];
                tmp['axislabel'+str(PC[0])] = 'PC' + str(PC[0]);
                tmp['score_' + str(PC[1])] = data_u[PC[1]][cnt]['u_matrix'];
                tmp['axislabel'+str(PC[1])] = 'PC' + str(PC[1]);
                del tmp['u_matrix'];
                del tmp['singular_value_index'];
                data_u_O[str(PC)].append(tmp);
            #extract out the loadings
            data_v_O[str(PC)]=[];
            for cnt,d in enumerate(data_v[PC[0]][:]):
                if d['component_name'] != data_v[PC[1]][cnt]['component_name'] and d['calculated_concentration_units'] != data_v[PC[1]][cnt]['calculated_concentration_units']:
                    print('data is not in the correct order');
                tmp = copy.copy(d);
                tmp['loadings_' + str(PC[0])] = d['v_matrix'];
                tmp['axislabel'+str(PC[0])] = 'Loadings' + str(PC[0]);
                tmp['loadings_' + str(PC[1])] = data_v[PC[1]][cnt]['v_matrix'];
                tmp['axislabel'+str(PC[1])] = 'Loadings' + str(PC[1]);
                del tmp['v_matrix'];
                del tmp['singular_value_index'];
                data_v_O[str(PC)].append(tmp);
        return data_u_O,data_v_O;
    def extract_UAndVMatrices_2D_byPCAndMethod(self,data_u,data_v,
            PCs,methods,
            score_column_I = 'u_matrix',
            loadings_column_I = 'v_matrix',
            method_column_I='svd_method'):
        '''Extract out the scores and loadings
        INPUT:
        data_u = listDict of pca/pls scores
        data_v = listDict of pca/pls loadings
        PCs = [[],[],...] of integers, describing the 2D PC plots
            E.G. PCs = [[1,1],[2,2],[3,3]];
        methods = [[],[],...] of strings, describing the 2D plots
            E.G. methods = [['svd','robustSvd'],['svd','robustSvd'],[['svd','robustSvd']];
        OUTPUT:
        data_u_O = {'[1_svd,1_robustSvd]':[],'[2,2]':[],'[3,3]':[],...} where each [] is a listDict of the data from PCs e.g. 1,2
        data_v_O = {'[1_svd,1_robustSvd]':[],'[2,2]':[],'[3,3]':[],...}
        PCs_O = [[],[],...]
            E.G. [[1_svd,1_robustSvd],[2_svd,2_robustSvd],[3_svd,3_robustSvd],..]
        '''
        PCs_O = [];
        data_u_O,data_v_O = {},{};
        for PC_cnt,PC in enumerate(PCs):
            #extract out the scores
            pc0 = str(PC[0])+'_'+methods[PC_cnt][0];
            pc1 = str(PC[1])+'_'+methods[PC_cnt][1];
            pc_list = [pc0,pc1];
            PCs_O.append(pc_list);
            data_u_O[str(pc_list)]=[];
            for cnt1,d1 in enumerate(data_u[PC[0]][:]):
                for cnt2,d2 in enumerate(data_u[PC[1]][:]):
                    if d1['sample_name_short'] == d2['sample_name_short'] \
                        and d1['calculated_concentration_units'] == d2['calculated_concentration_units'] \
                        and d1[method_column_I] == methods[PC_cnt][0] \
                        and d2[method_column_I] == methods[PC_cnt][1] :
                            tmp = copy.copy(d2);
                            tmp['score_' + pc0] = d1[score_column_I];
                            tmp['axislabel'+pc0] = 'PC' + pc0;
                            tmp['score_' + pc1] = d2[score_column_I];
                            tmp['axislabel'+pc1] = 'PC' + pc1;
                            del tmp['u_matrix'];
                            del tmp[method_column_I];
                            del tmp['singular_value_index'];
                            data_u_O[str(pc_list)].append(tmp);
                            break;
            #extract out the loadings
            data_v_O[str(pc_list)]=[];
            for cnt1,d1 in enumerate(data_v[PC[0]][:]):
                for cnt2,d2 in enumerate(data_v[PC[1]][:]):
                    if d1['component_name'] == d2['component_name'] \
                        and d1['calculated_concentration_units'] == d2['calculated_concentration_units'] \
                        and d1[method_column_I] == methods[PC_cnt][0] \
                        and d2[method_column_I] == methods[PC_cnt][1] :
                            tmp = copy.copy(d2);
                            tmp['loadings_' + pc0] = d1[loadings_column_I];
                            tmp['axislabel'+pc0] = 'Loadings' + pc0;
                            tmp['loadings_' + pc1] = d2[loadings_column_I];
                            tmp['axislabel'+pc1] = 'Loadings' + pc1;
                            del tmp['v_matrix'];
                            del tmp[method_column_I];
                            del tmp['singular_value_index'];
                            data_v_O[str(pc_list)].append(tmp);
                            break;
        return data_u_O,data_v_O,PCs_O;

