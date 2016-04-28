from .calculate_dependencies import *
from .calculate_base import calculate_base

class calculate_enrichment(calculate_base):
    
    def calculate_enrichment(self,enrichment_class_matrix_I, components_I):
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