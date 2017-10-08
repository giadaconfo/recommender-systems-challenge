import pandas as pd
import numpy as np
import os.path
from scipy.sparse import coo_matrix, save_npz
pd.set_option('display.max_columns',500)
os.chdir('/Users/LucaButera/git/rschallenge') #modify this according to your environment

#auxiliary
def get_sparse_index_val(couples, prim_index, sec_index):
    aux = couples.dropna(axis=0, how='any')
    return prim_index.loc[aux.iloc[:,0].values].values, sec_index.loc[aux.iloc[:,1].values].values

#Creates the ICM
def create_ICM(tracks):
    #auxiliary vars
    tg = ['tag1','tag2','tag3','tag4','tag5']
    sel = ['artist_id', 'album'] + tg

    items = np.unique(tracks['track_id'].values)
    artists = np.unique(tracks['artist_id'].values)
    albums = np.unique(tracks['album'].values)
    tags = np.unique(tracks[tg].values)

    #Structure to index the sparse ICM matrix using ids
    ICM_items = pd.Series(range(items.size), index=items)
    ICM_artists = pd.Series(range(artists.size), index=artists)
    ICM_albums = pd.Series(range(artists.size, artists.size + albums.size), index=albums)
    ICM_tags = pd.Series(range(artists.size + albums.size, artists.size + albums.size + tags.size), index=tags)

    #Creating ICM
    rows = np.array([], dtype='int32')
    columns = np.array([], dtype='int32')
    indexes = [ICM_artists,ICM_albums]+[ICM_tags]*5
    for label, index in zip(sel, indexes):
        tmp_c, tmp_r = get_sparse_index_val(tr_info[['track_id',label]], ICM_items, index)
        rows = np.append(rows, tmp_r)
        columns = np.append(columns, tmp_c)

    data = np.array([1]*len(rows), dtype='int32')

    ICM = coo_matrix((data,(rows,columns)), shape=(artists.size + albums.size + tags.size, items.size))

    return ICM, ICM_items, ICM_artists, ICM_albums, ICM_tags

#Importing the data
tr_info = pd.read_csv('Data/fixed_tracks_final.csv')

#Creating ICM plus auxiliary structures
ICM, ICM_items, ICM_artists, ICM_albums, ICM_tags = create_ICM(tr_info)

#Saving everything
save_npz('BuiltStructures/ICM.npz',ICM)
ICM_items.to_csv('BuiltStructures/ICM_items.csv')
ICM_artists.to_csv('BuiltStructures/ICM_artists.csv')
ICM_albums.to_csv('BuiltStructures/ICM_albums.csv')
ICM_tags.to_csv('BuiltStructures/ICM_tags.csv')

print(ICM.get_shape)
