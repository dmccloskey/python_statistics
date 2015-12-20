from .calculate_dependencies import *
from collections import Counter
from .calculate_base import calculate_base

class calculate_count(calculate_base):
    
    # counts and reducers
    def count_elements(self,data_I):
        '''count the number of occurances of a elements
        INPUT:
        data_I = list of data with multiple elements
        OUTPUT:
        elements_unique_O = list of unique elements
        elements_count_O = list of feature counts
        elements_count_fraction_O = list of feature counts expressed as a fraction of the total
        '''
        cnt = Counter();
        #count the occurances of the feature
        for d in data_I:
            cnt[d] += 1;
        #extract the unique features and counts
        elements_unique_O = [];
        elements_count_O = [];
        elements_count_fraction_O = [];
        count_sum = sum(cnt.values());
        for element in list(cnt):
            elements_unique_O.append(element);
            elements_count_O.append(cnt[element]);
            elements_count_fraction_O.append(cnt[element]/count_sum);
        return elements_unique_O,elements_count_O,elements_count_fraction_O;
