import numpy as np
import pandas as pd
import recsys as rs
import json
import notipy
import TopSimilarRecommender as TSR
import os.path
os.chdir('/Users/LucaButera/git/rschallenge')

dataset = pd.read_csv('Data/train_final.csv','\t')
tr_info = pd.read_csv('Data/tracks_final.csv','\t')

train, test, tgt_tracks, tgt_playlists = plit_train_test(dataset, 10, 20, 5, 2517):

fit_dict = {'tracks_info' : tr_info,
            'attributes' : ['artist_id', 'album', 'tags'],
            'attributes_to_prune' : ['tags'],
            'tgt_tracks' : tgt_tracks,
            'n_min_attr' : 2,
            'measure' : 'dot',
            'shrinkage' : 0,
            'n_el_sim' : 20}

recommend_dict = {'tgt_playlists' : tgt_playlists,
                  'train_playlists_tracks_pairs' : train,
                  'normalize' : False}

rec = TSR.TopSimilarRecommender()
rec.fit(**fit_dict)
notipy.notify('Model fitted!')
print('Model fitted!')

recommendetions = rec.recommend(**recommend_dict)
notipy.notify('Recommending completed!')
print('Recommending completed!')

map_eval = rs.evaluate(recommendetions, test, 'MAP')
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

with open('runs_data.json', 'a') as fp:
    json.dump(run_data, fp, indent=2)
    fp.write('\n')
notipy.notify('Run data saved!')
print('Run data saved!')
