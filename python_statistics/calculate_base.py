from .calculate_dependencies import *
class calculate_base():
    def __init__(self):
        self.data=[];
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