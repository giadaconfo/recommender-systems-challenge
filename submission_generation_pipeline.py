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

fit_dict = {'tracks_info' : tr_info,
            'attributes' : ['artist_id', 'album', 'tags', 'playcount', 'duration'],
            'attributes_to_prune' : ['tags'],
            'tgt_tracks' : tgt_tr,
            'n_min_attr' : 2,
            'measure' : 'dot',
            'shrinkage' : 0,
            'n_el_sim' : 20}

recommend_dict = {'tgt_playlists' : tgt_pl,
                  'train_playlists_tracks_pairs' : train,
                  'normalize' : False}

rec = TSR.TopSimilarRecommender()
rec.fit(**fit_dict)
notipy.notify('Model fitted!')
print('Model fitted!')

recommendetions = rec.recommend(**recommend_dict)
notipy.notify('Recommending completed!')
print('Recommending completed!')

recommendetions.to_csv('Submissions/top_similar_submission_' + str(1234) + '.csv', index=False)
notipy.notify('Results saved as csv!')
print('Results saved as csv!')
