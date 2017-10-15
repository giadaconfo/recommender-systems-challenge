import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import math
import collections
import sys

def fix_tracks_format(df):
    data = df.copy()
    data['album'] = data['album'].apply(fix_albums)
    data['playcount'] = data['playcount'].apply(fix_playcounts)
    data['duration'] = data['duration'].apply(fix_durations)
    data[['tag1','tag2','tag3','tag4','tag5']] = data['tags'].apply(fix_tags)
    data.drop(labels='tags', axis=1, inplace=True)
    return data

def fix_albums(val):
    return int(val[1:-1]) if val != '[None]' and val != '[]' else float('NaN')

def fix_tags(val):
    return pd.Series(val[1:-1].split(', '), dtype='int') if val != '[None]' and val != '[]' else pd.Series([float('NaN')]*5)

def fix_playcounts(val):
    return 'hi_playcount' if val >= 8000 else float('NaN')

def fix_durations(val):
    return 'hi_duration' if val >= 340000 else float('NaN')

#Richiede fixed_tracks_final, target_playlists e target_tracks
#Restituisce 3 pandas.Series per indicizzare item, target items e target playlists, e una namedtuple con gli attributi
'''def create_sparse_indexes(tracks, playlists, tracks_torec):
    items = np.unique(tracks['track_id'].values)
    artists = np.unique(tracks['artist_id'].values)
    albums = np.unique(tracks['album'].values)
    tags = np.unique(tracks[['tag' + str(i) for i in range(1,6)]].values)
    rec_pl = np.unique(playlists['playlist_id'].values)
    rec_tr = np.unique(tracks_torec['track_id'].values)

    IX_tgt_playlists = pd.Series(range(rec_pl.size), index=rec_pl)
    IX_items = pd.Series(range(items.size), index=items)
    IX_tgt_items = pd.Series(range(rec_tr.size), index=rec_tr)
    IX_artists = pd.Series(range(artists.size), index=artists)
    IX_albums = pd.Series(range(artists.size, artists.size + albums.size), index=albums)
    IX_tags = pd.Series(range(artists.size + albums.size, artists.size + albums.size + tags.size), index=tags)

    SparseIndexes = collections.namedtuple('SparseIndexes', ['artists','albums','tags'])
    Indexes = SparseIndexes([IX_artists,IX_albums,IX_tags])

    return IX_items, IX_tgt_items, IX_tgt_playlists, Indexes'''

def create_sparse_indexes(tracks_info=None, playlists=None, tracks_reduced=None, attr_list=None):
    if tracks_info is not None:
        items = np.unique(tracks_info['track_id'].values)
        IX_items = pd.Series(range(items.size), index=items)
    else:
        IX_items=None

    if playlists is not None:
        rec_pl = np.unique(playlists['playlist_id'].values)
        IX_tgt_playlists = pd.Series(range(rec_pl.size), index=rec_pl)
    else:
        IX_tgt_playlists=None

    if tracks_reduced is not None:
        rec_tr = np.unique(tracks_reduced['track_id'].values)
        IX_tgt_items = pd.Series(range(rec_tr.size), index=rec_tr)
    else:
        IX_tgt_items=None

    if attr_list is not None:
        attributes = [[] for i in range(len(attr_list))]
        for i, a in zip(range(len(attr_list)), attr_list):
            if not a == 'tags':
                attributes[i] = np.unique(tracks_info[a].values)
        if 'tags' in attr_list:
            attributes[-1] = np.unique(tracks_info[['tag' + str(j) for j in range(1,6)]].values)
            attr_list.append(attr_list.pop(attr_list.index('tags')))
        bound = 0
        indexes = [[] for i in range(len(attr_list))]
        for i,attr in zip(range(len(attr_list)),attributes):
            indexes[i] = pd.Series(range(bound, bound + attr.size), index=attr)
            bound += attr.size
        SparseIndexes = collections.namedtuple('SparseIndexes', attr_list)
        Indexes = SparseIndexes(*indexes)
    else:
        Indexes=None

    return IX_items, IX_tgt_items, IX_tgt_playlists, Indexes

def create_ICM(tracks_info, IX_items, Indexes, n_min_attr=0):
    rows = np.array([], dtype='int32')
    columns = np.array([], dtype='int32')
    attributes = Indexes._fields if 'tags' not in Indexes._fields else [i for i in Indexes._fields if not i == 'tags'] + ['tag' + str(i) for i in range(1,6)]
    for label in attributes:
        ix_label = label if 'tag' not in label else 'tags'
        tmp_c, tmp_r = get_sparse_index_val(tracks_info[['track_id',label]], IX_items, getattr(Indexes,ix_label))
        rows = np.append(rows, tmp_r)
        columns = np.append(columns, tmp_c)

    data = np.array([1]*len(rows), dtype='int32')
    att_size = 0
    for attr in Indexes:
        att_size += attr.index.values.size

    ICM = sps.coo_matrix((data,(rows,columns)), shape=(att_size, IX_items.index.values.size))

    if n_min_attr >= 2:
        prune_useless(ICM, n_min_attr)

    return ICM


def get_sparse_index_val(couples, prim_index, sec_index):
    aux = couples.dropna(axis=0, how='any')
    return prim_index.loc[aux.iloc[:,0].values].values, sec_index.loc[aux.iloc[:,1].values].values

def prune_useless(mat, n_min_attr):
    mat = mat.tocsr()
    to_del = []

    print('Pruning attributes.')
    print('Number of attributes before pruning is: ' + str(mat.shape[0]))

    for i in range(mat.shape[0]):
        if mat[i,:].nnz < n_min_attr:
            to_del += [i - len(to_del)]
    for i in to_del:
        delete_row_csr(mat, i)

    print('Number of attributes after pruning is: ' + str(mat.shape[0]))
    return

def delete_row_csr(mat, i):
    if not isinstance(mat, sps.csr_matrix):
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
    return

#Requires the sparse index for target playlists, for items and the dataset with playlists/tracks couples
def create_tgt_URM(IX_tgt_playlists, IX_items, playlist_to_track):
    rows = np.array([], dtype='int32')
    columns = np.array([], dtype='int32')
    for p in IX_tgt_playlists.index.values:
        tracks = playlist_to_track[playlist_to_track['playlist_id'] == p]['track_id'].values.astype('int32')
        rows = np.append(rows, np.array([IX_tgt_playlists.loc[p]]*tracks.size,dtype='int32'))
        columns = np.append(columns, IX_items.loc[tracks])
        if (IX_tgt_playlists.loc[p] % 1000 == 0):
            print('Calculated ' + str(IX_tgt_playlists.loc[p]) + ' users ratings over ' + str(IX_tgt_playlists.index.shape[0]))

    data = np.array([1]*len(rows), dtype='int32')

    URM = sps.coo_matrix((data,(rows,columns)), shape=(IX_tgt_playlists.index.shape[0], IX_items.shape[0]))
    return URM

def calculate_dot(ICM_i, rec_ICM, shrinkage=0):
    return ICM_i.T.dot(rec_ICM).toarray().flatten()

def calculate_cos(ICM_i, rec_ICM, shrinkage=0):
    dot = ICM_i.T.dot(rec_ICM).toarray().ravel()
    i_module = math.sqrt(np.sum([i**2 for i in ICM_i.toarray().ravel()]))
    ICM_modules = np.sqrt(rec_ICM.copy().power(2).sum(axis=1).toarray().ravel())
    cos = np.divide(dot, ICM_modules * i_module + shrinkage)
    return cos

def create_Smatrix(ICM, n_el=20, measure='dot',shrinkage=0, IX_tgt_items=None, IX_items=None):
    if ((IX_tgt_items is not None and IX_items is None) or (IX_tgt_items is None and IX_items is not None)):
        sys.exit('Error: IX_items and IX_tgt_items must be both None or both defined')

    Measures = collections.namedtuple('Measures', ['dot', 'cos'])
    SimMeasures = Measures(*[calculate_dot, calculate_cos])

    data = np.array([],dtype='float32')
    rows = np.array([],dtype='int32')
    columns = np.array([],dtype='int32')
    l = ICM.shape[1]

    ICM = ICM.tocsc()
    if (IX_tgt_items is not None and IX_items is not None):
        rec_ICM = ICM[:,IX_items.loc[IX_tgt_items.index.values].values.flatten()]
        rec_ICM = rec_ICM.tocsc()
        h = IX_tgt_items.index.values.size
    else:
        rec_ICM = ICM
        h = l

    for i in range(l):
        sim = getattr(SimMeasures, measure)(ICM[:,i], rec_ICM, shrinkage)
        if (IX_tgt_items is None and IX_items is None):
            sim[i] = 0
        #NOT WORKING; YET TO DISCOVER WHY
        #elif (IX_tgt_items is not None and IX_items is not None and IX_items.index.values[i] in IX_tgt_items.index.values):
            #sim[IX_tgt_items.loc[IX_items.index.values[i]]] = 0
            #print('Diagonal to 0 at iteration #' + str(i))

        sort = np.argsort(sim)[-n_el:].astype(np.int32)
        data = np.append(data, sim[sort])
        rows = np.append(rows, np.array([i]*n_el,dtype='int32'))
        columns = np.append(columns, sort)
        if (i % 1000 == 0):
            print('Computed ' + str(i) + ' similarities over ' + str(l) + ' with ' + measure + ' measure and ' + str(shrinkage) + ' shrinkage.')

    S = sps.coo_matrix((data,(rows,columns)), shape=(l, h))
    return S

def top5_outside_playlist(ratings, p_id, train_playlists_tracks_pairs, IX_tgt_playlists, IX_tgt_items):
    tgt_in_playlist = np.intersect1d(train_playlists_tracks_pairs[train_playlists_tracks_pairs['playlist_id'] == IX_tgt_playlists.index.values[p_id]]['track_id'].values, IX_tgt_items.index.values, assume_unique=True)
    ratings[IX_tgt_items.loc[tgt_in_playlist].values] = 0 #line to change

    if(np.count_nonzero(ratings) < 5): sys.exit('Not enough similarity')

    top5_ind = np.flip(np.argsort(ratings)[-5:], axis=0) #Contains the index of the recommended songs
    return IX_tgt_items.index.values[top5_ind]

def sub_format(l):
    res = " ".join(np.array_str(l).split())[1:-1]
    return res
'''Requires:
    The dataset generated from the recommender system
    The test set
    A string indicating the metric for evaluation'''
def evaluate(results, test, eval_metric='MAP'):
    if eval_metric == 'MAP':
        APs = results.apply(calculate_AP, args=test)
        res = (APs.sum())/results.shape[0]
    return res

def calculate_AP(row, test):
    p_id = row['playlist_id'].values[0]
    recs = np.fromstring(row['track_ids'].values[0], dtype=float, sep=' ')

    AP = 0
    rel_sum = 0
    n_rel_items = min(test[test['playlist_id'] == p_id].shape[0],5)
    for i in range(recs.size):
        rel = 1 if ((test['playlist_id'] == p_id) & (test['track_id'] == recs[i])).any() else 0
        rel_sum += rel
        P = rel_sum/i+1
        AP += (P * rel)/n_rel_items

    return AP
