import numpy as np
import os.path
from scipy.sparse import *
os.chdir('/home/giada/github/RecSys') #modify this according to your environment

#Loading pruned ICM
ICM = load_npz('BuiltStructures/prunedICM.npz').tocsc()

data = np.array([],dtype='int32')
rows = np.array([],dtype='int32')
columns = np.array([],dtype='int32')
l = ICM.shape[1]
n_el = 100

for i in range(l):
    dot = ICM[:,i].T.dot(ICM).toarray().flatten()
    dot[i] = 0
    sort = np.argsort(dot)[-n_el:].astype(np.int32)
    data = np.append(data, dot[sort])
    rows = np.append(rows, np.array([i]*n_el,dtype='int32'))
    columns = np.append(columns, sort)
    print(i)

S = coo_matrix((data,(rows,columns)), shape=(l, l))

save_npz('BuiltStructures/SimilarityMatrix.npz',S)

print(S.get_shape)
