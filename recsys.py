import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
from scipy.sparse import linalg as la
import math
import collections
import sys
import ctypes
import random
from tqdm import tqdm
from sklearn import preprocessing as prp

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
    return 1 if val >= 8000 else float('NaN')

def fix_durations(val):
    return 1 if val >= 340000 else float('NaN')

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
        items = tracks_info['track_id'].dropna().unique()
        IX_items = pd.Series(range(items.size), index=items)
    else:
        IX_items=None

    if playlists is not None:
        rec_pl = playlists['playlist_id'].dropna().unique()
        IX_tgt_playlists = pd.Series(range(rec_pl.size), index=rec_pl)
    else:
        IX_tgt_playlists=None

    if tracks_reduced is not None:
        rec_tr = tracks_reduced['track_id'].dropna().unique()
        IX_tgt_items = pd.Series(range(rec_tr.size), index=rec_tr)
    else:
        IX_tgt_items=None

    if attr_list is not None:
        attributes = [[] for i in range(len(attr_list))]
        for i, a in zip(range(len([i for i in attr_list if not i == 'tags'])), [i for i in attr_list if not i == 'tags']):
            attributes[i] = tracks_info[a].dropna().unique()
        if 'tags' in attr_list:
            attributes[-1] = np.unique(tracks_info[['tag' + str(j) for j in range(1,6)]].values)
            attributes[-1] = attributes[-1][~np.isnan(attributes[-1])]
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

def create_ICM(tracks_info, IX_items, Indexes, attr_list):
    rows = np.array([], dtype='int32')
    columns = np.array([], dtype='int32')
    attributes = attr_list if 'tags' not in attr_list else [i for i in attr_list if not i == 'tags'] + ['tag' + str(i) for i in range(1,6)]
    for label in attributes:
        ix_label = label if 'tag' not in label else 'tags'
        tmp_c, tmp_r = get_sparse_index_val(tracks_info[['track_id',label]], IX_items, getattr(Indexes,ix_label))
        rows = np.append(rows, tmp_r)
        columns = np.append(columns, tmp_c)

    data = np.array([1]*len(rows), dtype='int32')
    att_size = 0
    for attr in Indexes:
        att_size += attr.index.values.size

    ICM = sps.coo_matrix((data,(rows,columns)), shape=(att_size, IX_items.shape[0]))

    return ICM

def get_sparse_index_val(couples, prim_index, sec_index):
    aux = couples.dropna(axis=0, how='any')
    return prim_index.loc[aux.iloc[:,0].values].values, sec_index.loc[aux.iloc[:,1].values].values

'''
#deprecated
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

#deprecated
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
'''

def delete_low_frequency_attributes(dataset, attributes, n_min):
    for a in attributes:
        if a == 'tags':
            n_appearance = pd.Series(dataset[['tag' + str(i) for i in range(1,6)]].values.ravel('F')).value_counts()
        else:
            n_appearance = dataset[a].value_counts()
        to_del = n_appearance[n_appearance < n_min].index.values
        if a == 'tags':
            for j in ['tag' + str(i) for i in range(1,6)]:
                dataset[j] = dataset[j].apply(lambda x: float('NaN') if x in to_del else x)
        else:
            dataset[a] = dataset[a].apply(lambda x: float('NaN') if x in to_del else x)
    return dataset

#Requires the sparse index for target playlists, for items and the dataset with playlists/tracks couples
def create_tgt_URM(IX_tgt_playlists, IX_items, playlist_to_track):
    rows = np.array([], dtype='int32')
    columns = np.array([], dtype='int32')
    for p in tqdm(IX_tgt_playlists.index.values):
        tracks = playlist_to_track[playlist_to_track['playlist_id'] == p]['track_id'].values.astype('int32')
        rows = np.append(rows, np.array([IX_tgt_playlists.loc[p]]*tracks.size,dtype='int32'))
        columns = np.append(columns, IX_items.loc[tracks])
    data = np.array([1]*len(rows), dtype='int32')

    URM = sps.coo_matrix((data,(rows,columns)), shape=(IX_tgt_playlists.index.shape[0], IX_items.shape[0]))
    return URM

def create_UBR_URM(IX_playlists, IX_tgt_items, train):
    rows = np.array([], dtype='int32')
    columns = np.array([], dtype='int32')
    for p in tqdm(IX_playlists.index.values):
        tracks = train[train['playlist_id'] == p]['track_id'].values.astype('int32')
        tracks = tracks[np.in1d(tracks, IX_tgt_items.index)]
        rows = np.append(rows, np.array([IX_playlists.loc[p]]*tracks.size,dtype='int32'))
        columns = np.append(columns, IX_tgt_items.loc[tracks])
    data = np.array([1]*len(rows), dtype='int32')

    URM = sps.coo_matrix((data,(rows,columns)), shape=(IX_playlists.index.shape[0], IX_tgt_items.shape[0]))
    return URM

def calculate_dot(ICM_i, rec_ICM, shrinkage=0):
    return ICM_i.T.dot(rec_ICM).toarray().flatten()

def calculate_cos(ICM_i, rec_ICM, shrinkage=0):
    dot = ICM_i.T.dot(rec_ICM).toarray().ravel()
    i_module = math.sqrt(np.sum([i**2 for i in ICM_i.toarray().ravel()]))
    ICM_modules = np.asarray(np.sqrt(rec_ICM.copy().power(2).sum(axis=0))).ravel()
    cos = np.divide(dot, ICM_modules * i_module + shrinkage)
    return cos

def calculate_prob(ICM_i, rec_ICM, shrinkage=0):
    dot = ICM_i.T.dot(rec_ICM).toarray().ravel()
    ICM_modules = np.asarray(rec_ICM.sum(axis=0)).ravel()
    prob = np.divide(dot, ICM_modules + shrinkage)
    return prob

def calculate_implicit_cos(ICM_i, rec_ICM, shrinkage=0):
    dot = ICM_i.T.dot(rec_ICM).toarray().ravel()
    i_module = math.sqrt(ICM_i.sum(axis=0)[0,0])
    ICM_modules = np.asarray(np.sqrt(rec_ICM.sum(axis=0))).ravel()
    imp_cos = np.divide(dot, ICM_modules * i_module + shrinkage)
    return imp_cos

def create_Smatrix(ICM, n_el=20, measure='dot',shrinkage=0, IX_tgt_items=None, IX_items=None):
    if ((IX_tgt_items is not None and IX_items is None) or (IX_tgt_items is None and IX_items is not None)):
        sys.exit('Error: IX_items and IX_tgt_items must be both None or both defined')

    Measures = collections.namedtuple('Measures', ['dot', 'cos', 'prob', 'imp_cos'])
    SimMeasures = Measures(*[calculate_dot, calculate_cos, calculate_prob, calculate_implicit_cos])

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

    for i in tqdm(range(l)):
        sim = getattr(SimMeasures, measure)(ICM[:,i], rec_ICM, shrinkage)
        if (IX_tgt_items is None and IX_items is None):
            sim[i] = 0
        #SEEMS TO WORK, KEEP AN EYE ON IT!
        elif (IX_tgt_items is not None and IX_items is not None and IX_items.index.values[i] in IX_tgt_items.index.values):
            sim[IX_tgt_items.loc[IX_items.index.values[i]]] = 0
            #print('Diagonal to 0 at iteration #' + str(i))

        sort = np.argsort(sim)[-n_el:].astype(np.int32)
        data = np.append(data, sim[sort])
        rows = np.append(rows, np.array([i]*n_el,dtype='int32'))
        columns = np.append(columns, sort)
        
    S = sps.coo_matrix((data,(rows,columns)), shape=(l, h))
    return S
'''TO BE FIXED
def top5_outside_playlist(ratings, p_id, train_playlists_tracks_pairs, IX_tgt_playlists, IX_tgt_items, sim_check, secondary_sorting):
    tgt_in_playlist = np.intersect1d(train_playlists_tracks_pairs[train_playlists_tracks_pairs['playlist_id'] == IX_tgt_playlists.index.values[p_id]]['track_id'].values, IX_tgt_items.index.values, assume_unique=True)
    ratings[IX_tgt_items.loc[tgt_in_playlist].values] = 0 #line to change

    if((np.count_nonzero(ratings) < 5) and sim_check):
        sys.exit('Not enough similarity')

    if(secondary_sorting):
        treshold = np.argsort(ratings)[-5:-4]
        top5_id = sort_equal_by_popularity(ratings, treshold, train_playlists_tracks_pairs, IX_tgt_items)
    else:
        top5_ind = np.flip(np.argsort(ratings)[-5:], axis=0)
        top5_id = IX_tgt_items.index.values[top5_ind]

    return top5_id

def sort_equal_by_popularity(ratings, treshold, train, IX_tgt_items):
    competitors = IX_tgt_items[ratings >= treshold].index.values
    most_popular = train['track_id'][train['track_id'].isin(competitors)].value_counts()
    most_rated = np.flip(np.argsort(ratings), axis=0)
    most_rated = most_rated[ratings[most_rated] >= treshold]
    combined_type = np.dtype([('item', 'i4'), ('rate', 'f4'), ('pop', 'i4')])
    combined = np.empty(competitors.shape[0], combined_type)
    combined['item'] = competitors
    combined['rate'] = ratings[IX_tgt_items.loc[competitors]]
    combined['pop'] = most_popular.loc[competitors]
    ordered_ix = np.flip(np.argsort(combined, order=['rate', 'pop'])[-5:], axis=0)
    return combined['item'][ordered_ix]
'''

def top5_outside_playlist(ratings, p_id, train_playlists_tracks_pairs, IX_tgt_playlists, IX_tgt_items, sim_check, secondary_sorting):
    tgt_in_playlist = np.intersect1d(train_playlists_tracks_pairs[train_playlists_tracks_pairs['playlist_id'] == IX_tgt_playlists.index.values[p_id]]['track_id'].values, IX_tgt_items.index.values, assume_unique=True)
    ratings[IX_tgt_items.loc[tgt_in_playlist].values] = 0 #line to change

    #REMEMBER TO UNCOMMENT
    if((np.count_nonzero(ratings) < 5) and sim_check):
        sys.exit('Not enough similarity')

    top5_ind = np.flip(np.argsort(ratings)[-5:], axis=0) #Contains the index of the recommended songs

    if ratings[ratings >= ratings[top5_ind[-1]]].shape[0] > 5:
        top5_ind = break_equalities_by_popularity(ratings, top5_ind, train_playlists_tracks_pairs, IX_tgt_items)

    return IX_tgt_items.index.values[top5_ind]

def break_equalities_by_popularity(ratings, top5_ind, train, IX_tgt_items):
    competition_treshold = ratings[top5_ind[-1]]
    competitors_mask = ratings == competition_treshold
    n_open_positions = ratings[top5_ind][ratings[top5_ind] == competition_treshold].shape[0]
    competitors = IX_tgt_items[competitors_mask].index.values
    winners = train['track_id'][train['track_id'].isin(competitors)].value_counts().index.values[:n_open_positions]
    return np.append(top5_ind[:5 - n_open_positions], IX_tgt_items.loc[winners])

def sub_format(l):
    res = " ".join(np.array_str(l).split())[1:-1]
    return res

'''Requires:
    The dataset generated from the recommender system
    The test set
    A string indicating the metric for evaluation'''
def evaluate(results, test, eval_metric='MAP'):
    if eval_metric == 'MAP':
        APs = results.apply(calculate_AP, axis=1, args=(test,))
        res = (APs.sum())/results.shape[0]
    return res

def calculate_AP(row, test):
    p_id = row['playlist_id']
    recs = np.fromstring(row['track_ids'], dtype=float, sep=' ')

    AP = 0
    rel_sum = 0
    n_rel_items = min(test[test['playlist_id'] == p_id].shape[0],5)
    for i in range(recs.size):
        rel = 1 if ((test['playlist_id'] == int(p_id)) & (test['track_id'] == recs[i])).any() else 0
        rel_sum += rel
        P = rel_sum/(i+1)
        AP += (P * rel)/n_rel_items

    return AP

def split_train_test(track_playlist_couples, min_tracks_in_playlist=10, test_percentage=20, n_tracks_toremove=5, seed=2517):
    random.seed(seed)

    n_track_counts = track_playlist_couples['playlist_id'].value_counts()
    big_target_playlist = n_track_counts[n_track_counts >= min_tracks_in_playlist].index.values

    n_playlists_in_test = int((big_target_playlist.size*test_percentage)/100)
    tgt_playlists = pd.DataFrame({"playlist_id": random.sample(set(big_target_playlist), n_playlists_in_test)})

    tracks_in_test_pl = track_playlist_couples[track_playlist_couples['playlist_id'].isin(tgt_playlists['playlist_id'])]

    test = pd.DataFrame(columns=['playlist_id', 'track_id'], dtype='int32')
    indexes_to_remove = np.array([], dtype='int32')
    for i in tracks_in_test_pl['playlist_id'].unique():
        tmp = (tracks_in_test_pl.where(tracks_in_test_pl['playlist_id'] == i).dropna().sample(n_tracks_toremove)).index
        test = test.append(tracks_in_test_pl.loc[tmp,:]).astype('int64')
        indexes_to_remove = np.append(indexes_to_remove, values=tmp)

    train = tracks_in_test_pl.drop(indexes_to_remove)
    tgt_tracks = test.drop_duplicates('track_id')

    return train, test, tgt_tracks, tgt_playlists

def train_test_split_from_URM(interactions, min_interactions, split_count, fraction=None, random_state=None):
    """ Split recommendation data into train and test sets
        Params ------
        interactions : scipy.sparse matrix Interactions between users and items.
        split_count : int Number of user-item-interactions per user to move from training to test set.
        fractions : float Fraction of users to split off some of their interactions into test set. If None, then all users are considered. """

    train = interactions.copy().tocoo()
    test = sps.lil_matrix(train.shape)
    if (random_state):
        np.random.seed(random_state)

    if fraction:
        try:
            user_index = np.random.choice(np.where(np.bincount(train.row) >= min_interactions)[0], replace=False, size=np.int64(np.floor(fraction * train.shape[0]))).tolist()
        except:
            print(('Not enough users with > ' + str(min_interactions) + ' interactions for fraction of ' + str(fraction)))
            raise
    else:
        user_index = range(train.shape[0])

    train = train.tolil()
    interactions = interactions.tocsr()
    for user in user_index:
        test_interactions = np.random.choice(interactions.getrow(user).indices, size=split_count, replace=False)
        train[user, test_interactions] = 0.
        # These are just 1.0 right now
        test[user, test_interactions] = interactions[user, test_interactions]

    # Test and training are truly disjoint
    assert (train.multiply(test).nnz == 0)
    np.random.seed()

    return train.tocsr(), test.tocsr(), user_index

def train_test_split_interface(data, min_interactions=10, test_percentage=20, interactions_toremove=5, seed=2517):
    IX_items, _, IX_playlists, _ = create_sparse_indexes(tracks_info=data, playlists=data)
    interactions = create_tgt_URM(IX_playlists, IX_items, data)
    train_M, test_M, tgt_playlists_ix = train_test_split_from_URM(interactions, min_interactions, interactions_toremove, test_percentage/100, seed)

    tgt_playlists = pd.DataFrame({'playlist_id' : IX_playlists.index[tgt_playlists_ix]})
    train = pd.DataFrame({'playlist_id' : IX_playlists.index[train_M.nonzero()[0]], 'track_id' : IX_items.index[train_M.nonzero()[1]]})
    test = pd.DataFrame({'playlist_id' : IX_playlists.index[test_M.nonzero()[0]], 'track_id' : IX_items.index[test_M.nonzero()[1]]})
    tgt_tracks = pd.DataFrame({'track_id' : IX_items.index[np.unique(test_M.nonzero()[1])]})


    return train, test, tgt_tracks, tgt_playlists

def compute_idf(ICM):
    frequencies = np.asarray(ICM.sum(axis=1))
    n_items = ICM.shape[1]
    return np.log10(n_items / frequencies)

def ICM_idf_regularization(ICM):
    ICM = ICM.tocsr()
    idf = compute_idf(ICM)
    return ICM.multiply(np.broadcast_to(idf,shape=(ICM.shape)))

def implicit_weighted_ALS(URM, lambda_val=0.1, alpha=40, iterations=10, rank_size=20, seed=2517):
    conf_matrix = (alpha * URM)
    n_user = conf_matrix.shape[0]
    n_item = conf_matrix.shape[1]
    rand_init_state = np.random.RandomState(seed)

    X = sps.csr_matrix(rand_init_state.normal(size=(n_user, rank_size)))
    Y = sps.csr_matrix(rand_init_state.normal(size=(n_item, rank_size)))

    X_diag = sps.eye(n_user)
    Y_diag = sps.eye(n_item)
    lambda_diag = lambda_val * sps.eye(rank_size)

    for i_step in tqdm(range(iterations)):
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        for u in range(n_user):
            u_row = conf_matrix[u,:].toarray()
            preference_v = u_row.copy()
            preference_v[preference_v != 0] = 1
            CuI = sps.diags(u_row, [0])
            yTCuIY = Y.T.dot(CuI).dot(Y)
            yTCupu = Y.T.dot(CuI + Y_diag).dot(preference_v.T)
            X[u] = la.spsolve(yTy + yTCuIY + lambda_diag, yTCupu)

        for i in range(n_item):
            i_row = conf_matrix[:,i].T.toarray()
            preference_v = i_row.copy()
            preference_v[preference_v != 0] = 1
            CiI = sps.diags(i_row, [0])
            xTCiIX = X.T.dot(CiI).dot(X)
            xTCipi = X.T.dot(CiI + X_diag).dot(preference_v.T)
            Y[i] = la.spsolve(xTx + xTCiIX + lambda_diag, xTCipi)

    return X, Y.T

def merge_similarities(S1, S2, alpha):
    S1 = normalize_matrix(S1).tocsr()
    S2 = normalize_matrix(S2).tocsr()
    return (alpha*S1)+((1-alpha)*S2)

def normalize_matrix(M):
    M_zero_mean = remove_mean(M.tocsr())
    M_scaled = prp.scale(M_zero_mean.T.tocsc(), axis=0, with_mean=False).T
    return M_scaled

def remove_mean(M):
    tot = np.array(M.sum(axis=1).squeeze())[0]
    tot[tot == 0] = 1
    cts = np.diff(M.indptr)
    inverse_m = cts/tot
    m_len = inverse_m.shape[0]
    m = sps.spdiags(inverse_m, 0, m_len, m_len)
    return m*M

def generic_similarity_based_recommend(S, tracks, tgt_tracks, tgt_playlists, train_data, sim_check=True, secondary_sorting=True):
    IX_items, IX_tgt_items, IX_tgt_playlists, _ = create_sparse_indexes(tracks_info=tracks, tracks_reduced=tgt_tracks, playlists=tgt_playlists)
    URM = create_tgt_URM(IX_tgt_playlists, IX_items, train_data)
    URM = URM.tocsr()
    print('URM built')

    recommendetions = np.array([])
    for p in tqdm(IX_tgt_playlists.values):
        avg_sims = URM[p,:].dot(S).toarray().ravel()
        top = top5_outside_playlist(avg_sims, p, train_data, IX_tgt_playlists, IX_tgt_items, sim_check, secondary_sorting)
        recommendetions = np.append(recommendetions, sub_format(top))

    return pd.DataFrame({'playlist_id' : IX_tgt_playlists.index.values, 'track_ids' : recommendetions})
