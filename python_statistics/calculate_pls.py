from .calculate_dependencies import *
from .calculate_base import calculate_base
class calculate_pls(calculate_base):

    def calculate_plsda(self):
        '''calculate PLS-DA using PLS
        '''
        pass;

    def calculate_pls(self):
        '''calculate PLS'''
        pass;

    def calculate_pls_explainedVariance(self):
        '''calculate the explained variance of the pls scores'''
        pass;

    def calculate_pls_vip(self):
        '''calculate the variance of the pls scores and loadings'''
        pass;

    def convert_factorVector2responseMatrix(self,factors_I):
        '''convert a list of factors to a response matrix
        NOTES: required to convert pls to pls-da

        INPUT:
        factors_I = [] of strings
        
        OUTPUT:
        response_O = binary matrix of shape len(factor_I),len(factor_unique)
            where a 1 indicates association with the factor specified by that column
            and a 0 indicates no association with the factor specified by that column
        factors_O = list of unique factors of (the column names of response_O)
        '''
        pass;

