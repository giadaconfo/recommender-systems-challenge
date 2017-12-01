import pandas as pd
import numpy as np
import os.path
import recsys as rs
import notipy
import math
import TopSimilarRecommender as TSR
from scipy import sparse as sps
import random
os.chdir('/home/giada/github/RecSys')

train = pd.read_csv('Data/train_final.csv','\t')
tr_info = pd.read_csv('Data/tracks_final.csv','\t')
pl_info = pd.read_csv('Data/playlists_final.csv','\t')
tgt_pl = pd.read_csv('Data/target_playlists.csv','\t')
tgt_tr = pd.read_csv('Data/target_tracks.csv','\t')

train, test, tgt_tracks, tgt_playlists = rs.split_train_test(train, 10, 20, 5, 2517)

rec = TSR.TopSimilarRecommender()
fit_dict = {'tracks_info' : tr_info,
            'attributes' : ['artist_id', 'album', 'tags', 'playcount'],
            'attributes_to_prune' : ['tags'],
            'tgt_tracks' : tgt_tracks,
            'n_min_attr' : 5,
            'idf' : True,
            'measure' : 'dot',
            'shrinkage' : 0,
            'n_el_sim' : 20}

rec.fit_without_matrix_builder(**fit_dict)
print('Model fitted!')

_, _, rec.IX_tgt_playlists, _ = rs.create_sparse_indexes(playlists=tgt_playlists)
URM = rs.create_tgt_URM(rec.IX_tgt_playlists, rec.IX_items, train)
URM = URM.tocsr()
print('URM built')

URM.tocsc()

data = np.array([],dtype='float32')
rows = np.array([],dtype='int32')
columns = np.array([],dtype='int32')
nitems =URM.shape[1]
n_el = 20
vec_pow = np.vectorize(math.pow)
vec_sqrt= np.vectorize(math.sqrt)
sum_i = np.array(vec_pow(URM.sum(axis=0),2)) #array containing the sum of the ratings for each item to the power of 2..
sum_i.shape
H = 10

for i in range (nitems):
    den = vec_sqrt(sum_i*sum_i[0,i]).flatten()
    res= (URM[:,i].T.dot(URM)).toarray().flatten()
    res = res/(den + H + 1e-6)
    sort = np.argsort(res)[-n_el:].astype(np.int32)
    data = np.append(data, res[sort])
    rows = np.append(rows, np.array([i]*n_el,dtype='int32'))
    columns = np.append(columns, sort)
    if i%1000 == 0:
        print(i)
item_similarity = sps.coo_matrix((data,(rows,columns)), shape=(nitems, nitems))

sps.save_npz('BuiltStructures/ItemSimilarityMatrix.npz',S)
print(S.get_shape)
