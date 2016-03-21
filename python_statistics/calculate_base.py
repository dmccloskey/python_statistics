from .calculate_dependencies import *
class calculate_base():
    def __init__(self,data_I = None, data_train_I = None, data_test_I = None,
                 data_model_I = None, random_state_I = None):
        if data_I:
            self.data=data_I;
        else: self.data = [];

        if data_train_I:
            self.data_train=data_train_I;
        else: self.data_train = []; #nsamples x nfeatures

        if data_test_I:
            self.data_test=data_test_I;
        else: self.data_test = [];
        
        if data_model_I:
            self.data_model=data_model_I;
        else: self.data_model = None;
        
        if random_state_I:
            self.random_state = npr.seed(random_state_I);
        else: self.random_state = npr.seed(0);
        
        if model_score_I:
            self.model_score=model_score_I;
        else: self.model_score = None;
        
        if feature_union_I:
            self.feature_union=feature_union_I;
        else: self.feature_union = None;
        
        if pipeline_I:
            self.pipeline=pipeline_I;
        else: self.pipeline = None;

    # other
    def null(self, A, eps=1e-6):
        u, s, vh = numpy.linalg.svd(A,full_matrices=1,compute_uv=1)
        null_rows = [];
        rank = numpy.linalg.matrix_rank(A)
        for i in range(A.shape[1]):
            if i<rank:
                null_rows.append(False);
            else:
                null_rows.append(True);
        null_space = scipy.compress(null_rows, vh, axis=0)
        return null_space.T