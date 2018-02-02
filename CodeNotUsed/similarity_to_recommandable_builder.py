import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
os.chdir('/Users/LucaButera/git/rschallenge')
#os.chdir('/home/giada/github/RecSys')

#Loading pruned ICM and tracks to recommend
rec_tr = pd.read_csv('Data/target_tracks.csv','\t')
ICM_items = pd.read_csv('BuiltStructures/ICM_items.csv', index_col=0, header=None, names=['track_id'])
ICM_items_swapped = pd.Series(ICM_items.index.values, index=ICM_items )
ICM = sps.load_npz('BuiltStructures/prunedICM.npz').tocsc()
rec_ICM = ICM[:,ICM_items.loc[rec_tr['track_id'].values].values.flatten()]

#New index for recommendable target_tracks
ICM_tgt_items = pd.Series(range(rec_tr.values.size), index=rec_tr['track_id'].values)

#Matrix construction
data = np.array([],dtype='int32')
rows = np.array([],dtype='int32')
columns = np.array([],dtype='int32')
l = ICM.shape[1]
n_el = 20

for i in range(l):
    dot = ICM[:,i].T.dot(rec_ICM).toarray().flatten()
    sort = np.argsort(dot)[-n_el:].astype(np.int32)
    data = np.append(data, dot[sort])
    rows = np.append(rows, np.array([i]*n_el,dtype='int32'))
    columns = np.append(columns, sort)
    print(i)

S = sps.coo_matrix((data,(rows,columns)), shape=(l, rec_tr.shape[0]))

ICM_tgt_items.to_csv('BuiltStructures/ICM_tgt_items.csv')
sps.save_npz('BuiltStructures/RecommendableSimilarityMatrix.npz',S)

print(S.get_shape)
