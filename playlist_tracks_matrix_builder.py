import numpy as np
import pandas as pd
import os.path
from scipy.sparse import *
os.chdir('/Users/LucaButera/git/rschallenge') #modify this according to your environment

ICM_items = pd.read_csv('BuiltStructures/ICM_items.csv', index_col=0, header=None, names=['track_id'])
ICM_tgt_items = pd.read_csv('BuiltStructures/ICM_tgt_items.csv', index_col=0, header=None, names=['track_id'])
S = load_npz('BuiltStructures/RecommendableSimilarityMatrix.npz')

pl = pd.read_csv('Data/train_final.csv','\t')
rec_tr = pd.read_csv('Data/target_tracks.csv','\t')
rec_pl = pd.read_csv('Data/target_playlists.csv','\t')

pl = pl[pl['playlist_id'].isin(rec_pl['playlist_id'])]
uniq_pl = np.unique(pl['playlist_id'].values)

INDEX_pl = pd.Series(range(uniq_pl.shape[0]), index=uniq_pl)

rows = np.array([], dtype='int32')
columns = np.array([], dtype='int32')
for p in uniq_pl:
    tracks = pl[pl['playlist_id'] == p]['track_id'].values.astype('int32')
    rows = np.append(rows, np.array([INDEX_pl.loc[p]]*tracks.size,dtype='int32'))
    columns = np.append(columns, ICM_items.loc[tracks])
    print(INDEX_pl.loc[p])
    print(rows)
    print(columns)

data = np.array([1]*len(rows), dtype='int32')

P_T = coo_matrix((data,(rows,columns)), shape=(pl['playlist_id'].values.shape[0], ICM_items.shape[0]))

INDEX_pl.to_csv('BuiltStructures/INDEX_pl.csv')
save_npz('BuiltStructures/PL_TR_MAT.npz',P_T)

print(P_T.get_shape)
