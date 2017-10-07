import numpy as np
import os.path
from scipy.sparse import *
os.chdir('/Users/LucaButera/git/rschallenge') #modify this according to your environment

#Loading pruned ICM
ICM = load_npz('BuiltStructures/prunedICM.npz').tocsc()

data = np.array([],dtype='int16')
rows = np.array([],dtype='int16')
columns = np.array([],dtype='int16')
l = ICM.shape[1]

for i in range(l): #rimuovere diagonale
    dot = ICM[:,i].T.dot(ICM)
    sort = np.argsort(dot)[:10]
    data = np.append(data, dot[sort])
    rows = np.append(rows, np.array([i]*10))
    columns = np.append(columns, sort)
    print(i)

S = coo_matrix((data,(rows,columns)), shape=(l, l))

save_npz('BuiltStructures/SimilarityMatrix.npz',S)

print(S.get_shape)
