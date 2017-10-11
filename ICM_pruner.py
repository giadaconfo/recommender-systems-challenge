import pandas as pd
import numpy as np
import os.path
from scipy.sparse import *
pd.set_option('display.max_columns',500)
os.chdir('/home/giada/github/RecSys') #modify this according to your environment

#Row deletion method
def delete_row_csr(mat, i):
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

#Prunes useless rows
def prune_useless(mat):
    to_del = []
    for i in range(mat.shape[0]):
        if mat[i,:].nnz == 1:
            to_del += [i - len(to_del)]
    for i in to_del:
        delete_row_csr(mat, i)

#Loading ICM
ICM = load_npz('BuiltStructures/ICM.npz').tocsr()

#Pruning
print('Number of elements before pruning is: ' + str(ICM.nnz))
prune_useless(ICM)
print('Number of elements after pruning is: ' + str(ICM.nnz))

#Saving pruned ICM
save_npz('BuiltStructures/prunedICM.npz', ICM)
print(ICM.get_shape)
