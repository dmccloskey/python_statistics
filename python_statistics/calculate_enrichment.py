from .calculate_dependencies import *
from .calculate_base import calculate_base

class calculate_enrichment(calculate_base):
    
    def calculate_enrichment_hypergeometric_v01(self,enrichment_class_matrix_I, components_I):
        """Calculate hypergoemotric enrichment of the set for each class

        The pathway matrix should have class in rows and components in columns
        """
        # only consider genes which are known to be in pathways
        class_component_list = components_I.intersection(enrichment_class_matrix_I.columns)
        # Generate hypergeometric distributions for each pathway. Each
        # pathway needs its own because they have different lenghts
        distributions = [hypergeom(len(enrichment_class_matrix_I.columns), l,
                                   len(class_component_list))
                         for l in enrichment_class_matrix_I.sum(axis=1)]
        class_hits = enrichment_class_matrix_I[class_component_list].sum(axis=1)
        # Each p-value for the hypergeometric enrichment is
        # survival function + 0.5 * pmf
        significance = [dist.sf(x) + 0.5 * dist.pmf(x)
                        for x, dist in zip(class_hits, distributions)]
        return Series(significance, index=enrichment_class_matrix_I.index);
    
    def calculate_enrichment_hypergeometric(
        self,
        enrichment_class_matrix_I,
        enrichment_classes_I, 
        enrichment_components_I, 
        components_I,
        use_weights_I=True):
        """Calculate hypergoemotric enrichment of the set for each class
        INPUT:
        enrichment_class_matrix_I = matrix where rows correspond to enrichment_classes
                                                columns correspond to enrichment_components
                                                 values correspond to enrichment_weights
        enrichment_classes_I = row labels of enrichment_class_matrix_I
        enrichment_components_I = column labels of enrichment_class_matrix_I
        components_I = list of significant components
        """
        pvalues_O = [];
        n_classes = len(enrichment_classes_I);
        n_components = len(enrichment_components_I);
        n_components_subset = len(components_I);
        for i,enrichment_class in enumerate(enrichment_classes_I):
            # Generate hypergeometric distributions for each pathway
            if use_weights_I:
                weights_all_sum = enrichment_class_matrix_I[i,:].sum();
                weights_subset_sum = sum([v for cnt,v in enumerate(enrichment_class_matrix_I[i,:]) if enrichment_components_I[cnt] in components_I])
            else:
                weights_all_sum = np.count_nonzero(enrichment_class_matrix_I[i,:])
                weights_subset_sum = np.count_nonzero([v for cnt,v in enumerate(enrichment_class_matrix_I[i,:]) if enrichment_components_I[cnt] in components_I])
            distribution = hypergeom(n_components,weights_all_sum,n_components_subset)
            # Each p-value for the hypergeometric enrichment is
            # survival function + 0.5 * pmf
            pvalue = distribution.sf(weights_subset_sum) + 0.5* distribution.pmf(weights_subset_sum);
            if np.isnan(pvalue):
                pvalue=1.0;
            pvalues_O.append(pvalue);
        return pvalues_O;