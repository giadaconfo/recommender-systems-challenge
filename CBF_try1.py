import pandas as pd
import numpy as np
import os.path
from scipy.sparse import coo_matrix
pd.set_option('display.max_columns',500)
os.chdir('/Users/LucaButera/git/rschallenge') #modify this according to your environment

#Creates the ICM
def createICM(tracks):
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
    rows = []
    columns = []
    for i,r in tr_info.set_index('track_id').iterrows():
        columns += [ICM_items[i]]*r[sel].count()
        if  r['artist_id'] != 'NaN': rows += [ICM_artists[r['artist_id']]]
        if r['album'] != 'NaN': rows += [ICM_albums[r['album']]]
        if r['tag1'] != 'NaN': rows += [ICM_tags[r['tag1']]] + [ICM_tags[r['tag2']]] + [ICM_tags[r['tag3']]] + [ICM_tags[r['tag4']]] + [ICM_tags[r['tag5']]]

    data = [1]*len(rows)

    ICM = coo_matrix(data,(rows,columns))

    return ICM, ICM_items, ICM_artists, ICM_albums, ICM_tags

#Importing the data
tr_info = pd.read_csv('Data/fixed_tracks_final.csv')

#Creating ICM plus auxiliary structures
ICM, ICM_items, ICM_artists, ICM_albums, ICM_tags = createICM(tr_info)

#Saving everything
save_npz('BuiltStructures/ICM/ICM.npz',ICM)
ICM_items.to_csv('BuiltStructures/ICM/ICM_items.csv')
ICM_artists.to_csv('BuiltStructures/ICM/ICM_artists.csv')
ICM_albums.to_csv('BuiltStructures/ICM/ICM_albums.csv')
ICM_tags.to_csv('BuiltStructures/ICM/ICM_tags.csv')
