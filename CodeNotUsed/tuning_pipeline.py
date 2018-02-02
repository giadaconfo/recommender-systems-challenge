import numpy as np
import pandas as pd
import recsys as rs
import json
import notipy
import time
import TopSimilarRecommender as TSR
import os.path
os.chdir('/Users/LucaButera/git/rschallenge')

dataset = pd.read_csv('Data/train_final.csv','\t')
tr_info = pd.read_csv('Data/tracks_final.csv','\t')

train, test, tgt_tracks, tgt_playlists = rs.split_train_test(dataset, 10, 20, 5, 2517)

fit_dict = {'tracks_info' : tr_info,
            'attributes' : ['artist_id', 'album', 'tags'],
            'attributes_to_prune' : ['tags'],
            'tgt_tracks' : tgt_tracks,
            'n_min_attr' : 90,
            'idf' : True,
            'measure' : 'dot',
            'shrinkage' : 0,
            'n_el_sim' : 15}

recommend_dict = {'tgt_playlists' : tgt_playlists,
                  'train_playlists_tracks_pairs' : train,
                  'normalize' : False}

rec = TSR.TopSimilarRecommender()

start = time.time()
rec.fit(**fit_dict)
end = time.time()
notipy.notify('Model fitted in ' + str(int(end - start)) + ' seconds!')
print('Model fitted in ' + str(int(end - start)) + ' seconds!')

start = time.time()
recommendetions = rec.recommend(**recommend_dict)
end = time.time()
notipy.notify('Recommending completed in ' + str(int(end - start)) + ' seconds!')
print('Recommending completed in ' + str(int(end - start)) + ' seconds!')

start = time.time()
map_eval = rs.evaluate(recommendetions, test, 'MAP')
end = time.time()
notipy.notify('Evaluation completed! MAP5 score is: ' + str(map_eval))
print('Evaluation completed! MAP5 score is: ' + str(map_eval))

run_data = {'recommender_type' : rec.__class__.__name__,
            'fit_parameters' : {'attributes' : fit_dict['attributes'],
                                'attributes_to_prune' : fit_dict['attributes_to_prune'],
                                'n_min_attr' : fit_dict['n_min_attr'],
                                'idf' : fit_dict['idf'],
                                'measure' : fit_dict['measure'],
                                'shrinkage' : fit_dict['shrinkage'],
                                'n_el_sim' : fit_dict['n_el_sim']},
            'recommend_parameters' : {'normalize' : recommend_dict['normalize']},
            'evaluation_result' : map_eval}

with open('runs_data.json', 'a') as fp:
    json.dump(run_data, fp, indent=4)
    fp.write('\n')
notipy.notify('Run data saved!')
print('Run data saved!')
