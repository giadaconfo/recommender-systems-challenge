import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import collections

def fix_tracks_format(data):
    data['album'] = data['album'].apply(fix_artists)
    data[['tag1','tag2','tag3','tag4','tag5']] = data['tags'].apply(fix_tags)
    data.drop(labels='tags', axis=1, inplace=True)
    return data

def fix_artists(val):
    return int(val[1:-1]) if val != '[None]' and val != '[]' else 'NaN'

def fix_tags(val):
    return pd.Series(val[1:-1].split(', '), dtype='int') if val != '[None]' and val != '[]' else pd.Series(['NaN']*5)

#Richiede fixed_tracks_final, target_playlists e target_tracks
#Restituisce 3 pandas.Series per indicizzare item, target items e target playlists, e una namedtuple con gli attributi
def create_sparse_indexes(tracks, playlists, tracks_torec):
    items = np.unique(tracks['track_id'].values)
    artists = np.unique(tracks['artist_id'].values)
    albums = np.unique(tracks['album'].values)
    tags = np.unique(tracks['tag' + str(i) for i in range(1,6)].values)
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

    return IX_items, IX_tgt_items, IX_tgt_playlists, Indexes

def create_ICM(attributes, IX_items, Indexes):
    rows = np.array([], dtype='int32')
    columns = np.array([], dtype='int32')
    for label in attributes:
        tmp_c, tmp_r = get_sparse_index_val(tr_info[['track_id',label]], IX_items, Indexes.label)
        rows = np.append(rows, tmp_r)
        columns = np.append(columns, tmp_c)

    data = np.array([1]*len(rows), dtype='int32')
    att_size = 0
    for field in Indexes._fields:
        att_size += Indexes.field.index.values.size

    ICM = sps.coo_matrix((data,(rows,columns)), shape=(att_size, IX_items.index.values.size))

def get_sparse_index_val(couples, prim_index, sec_index):
    aux = couples.dropna(axis=0, how='any')
    return prim_index.loc[aux.iloc[:,0].values].values, sec_index.loc[aux.iloc[:,1].values].values

def prune_useless(mat):
    mat.tocsr()
    to_del = []

    print('Number of elements before pruning is: ' + str(mat.nnz))

    for i in range(mat.shape[0]):
        if mat[i,:].nnz == 1:
            to_del += [i - len(to_del)]
    for i in to_del:
        delete_row_csr(mat, i)

    print('Number of elements after pruning is: ' + str(mat.nnz))
    return

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
    return

#Requires the sparse index for target playlists, for items and the dataset with playlists/tracks couples
def create_tgt_URM(IX_tgt_playlists, IX_items, playlist_to_track):
    rows = np.array([], dtype='int32')
    columns = np.array([], dtype='int32')
    for p in IX_tgt_playlists.index.values:
        tracks = playlist_to_track[playlist_to_track['playlist_id'] == p]['track_id'].values.astype('int32')
        rows = np.append(rows, np.array([IX_tgt_playlists.loc[p]]*tracks.size,dtype='int32'))
        columns = np.append(columns, IX_items.loc[tracks])
        print(INDEX_pl.loc[p])

    data = np.array([1]*len(rows), dtype='int32')

    URM = sps.coo_matrix((data,(rows,columns)), shape=(pl['playlist_id'].values.shape[0], ICM_items.shape[0]))
    return URM

def create_Smatrix(ICM, n_el=20, IX_tgt_items=None, IX_items=None):
    data = np.array([],dtype='int32')
    rows = np.array([],dtype='int32')
    columns = np.array([],dtype='int32')
    l = ICM.shape[1]

    if (IX_tgt_items is not None and IX_items is not None):
        rec_ICM = ICM[:,IX_items.loc[IX_tgt_items.index.values].values.flatten()]
        h = IX_tgt_items.index.values.size
    else:
        rec_ICM = ICM
        h = l

    for i in range(l):
        dot = ICM[:,i].T.dot(rec_ICM).toarray().flatten()
        if (IX_tgt_items is None or IX_items is None): dot[i] = 0
        sort = np.argsort(dot)[-n_el:].astype(np.int32)
        data = np.append(data, dot[sort])
        rows = np.append(rows, np.array([i]*n_el,dtype='int32'))
        columns = np.append(columns, sort)
        print(i)

    S = sps.coo_matrix((data,(rows,columns)), shape=(l, h))
    return S
