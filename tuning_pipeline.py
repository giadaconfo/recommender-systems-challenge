import numpy as np
import pandas as pd
import recsys as rs
import json
import notipy
import random
import TopSimilarRecommender as TSR
import os.path
os.chdir('/Users/LucaButera/git/rschallenge')

train = pd.read_csv('Data/train_final.csv','\t')
tr_info = pd.read_csv('Data/tracks_final.csv','\t')
tgt_pl = pd.read_csv('Data/target_playlists.csv','\t')
tgt_tr = pd.read_csv('Data/target_tracks.csv','\t')

big_target_playlist = [i for i in np.unique(train['playlist_id'].values) if train[train['playlist_id'] == i].shape[0] >= 10]

test_pl = pd.DataFrame({"playlist_id": list(map(lambda x: random.choice(big_target_playlist), range(100)))})

tracks_in_test_pl = pd.merge(train, test_pl, how='inner', on='playlist_id')
tracks_removed= pd.DataFrame(columns=['playlist_id', 'track_id'], dtype='int32')
indexes_to_remove= np.array([], dtype='int32')
for i in np.unique(tracks_in_test_pl['playlist_id'].values):
    #randomly selects 5 elements from playlist i and save indexes
    tmp = (tracks_in_test_pl.where(tracks_in_test_pl['playlist_id']==i).dropna().sample(5)).index
    #insert the items selected in tracks_removed dataFrame
    tracks_removed= tracks_removed.append(tracks_in_test_pl.take(tmp)).astype('int64')
    #remove tracks from the original dataframe
    indexes_to_remove=np.append(indexes_to_remove, values=tmp)

tracks_in_train= pd.DataFrame(tracks_in_test_pl)
tracks_in_train.drop(indexes_to_remove, inplace=True)

tgt_test_tracks= pd.DataFrame(tracks_in_test_pl['track_id'])
print(tgt_test_tracks.shape)
tgt_test_tracks = pd.DataFrame.drop_duplicates(tgt_test_tracks)
print(tgt_test_tracks.shape)

fit_dict = {'tracks_info' : tr_info,
            'attributes' : ['artist_id', 'album', 'tags'],
            'tgt_tracks' : tgt_test_tracks,
            'n_min_attr' : 2,
            'measure' : 'dot',
            'shrinkage' : 0,
            'n_el_sim' : 20}

recommend_dict = {'tgt_playlists' : test_pl,
                  'train_playlists_tracks_pairs' : tracks_in_test_pl,
                  'normalize' : False}

rec = TSR.TopSimilarRecommender()
rec.fit(**fit_dict)
notipy.notify('Model fitted!')
print('Model fitted!')

recommendetions = rec.recommend(**recommend_dict)
notipy.notify('Recommending completed!')
print('Recommending completed!')

map_eval = rs.evaluate(recommendetions, tracks_removed, 'MAP')
notipy.notify('Evaluation completed!')
print('Evaluation completed!')

run_data = {'recommender_type' : rec.__class__.__name__,
            'fit_parameters' : {'attributes' : fit_dict['attributes'],
                                'n_min_attr' : fit_dict['n_min_attr'],
                                'measure' : fit_dict['measure'],
                                'shrinkage' : fit_dict['shrinkage'],
                                'n_el_sim' : fit_dict['n_el_sim']},
            'recommend_parameters' : {'normalize' : recommend_dict['normalize']},
            'evaluation_result' : map_eval}

with open('runs_data.json', 'w') as fp:
    json.dump(run_data, fp)
notipy.notify('Run data saved!')
print('Run data saved!')
