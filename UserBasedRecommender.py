import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import recsys as rs
from tqdm import tqdm
os.chdir('/Users/LucaButera/git/rschallenge')
#os.chdir('/home/giada/github/RecSys')

class UserBasedRecommender:

    S = None
    IX_users =None
    IX_items = None
    IX_tgt_items = None

    def __init__(self, idf=False, measure='dot', shrinkage=0, n_el_sim=20):
        self.idf = idf
        self.measure = measure
        self.shrinkage = shrinkage
        self.n_el_sim = n_el_sim
        return


    def fit(self, track_ids, train_data, tgt_tracks=None):
        UserBasedRecommender.IX_items, UserBasedRecommender.IX_tgt_items, playlists_ix, _ = rs.create_sparse_indexes(tracks_info=track_ids, playlists=train_data, tracks_reduced=tgt_tracks)
        print('Calculated Indices')

        model_URM = rs.create_tgt_URM(playlists_ix, UserBasedRecommender.IX_items, train_data)
        model_URM = model_URM.tocsr()
        print(model_URM.shape)
        print('Model URM built')

        if self.idf:
            model_URM = rs.ICM_idf_regularization(model_URM)
            print('Model URM regularized with IDF!')
        print(model_URM.T.shape[1])
        UserBasedRecommender.S = rs.create_Smatrix(model_URM, self.n_el_sim, self.measure, self.shrinkage)
        print('Similarity built')
        print(UserBasedRecommender.S.shape)

    def recommend(self, tgt_playlists, train_data, sim_check=True, secondary_sorting=True):
        print(UserBasedRecommender.IX_items)
        _, _, IX_tgt_playlists, _ = rs.create_sparse_indexes(playlists=tgt_playlists)
        URM = rs.create_tgt_URM(IX_tgt_playlists, UserBasedRecommender.IX_items, train_data)
        URM = URM.tocsr()
        print(URM.shape)
        print('URM built')

        recommendetions = np.array([])
        for p in IX_tgt_playlists.values:
            avg_sims = URM[p,:].dot(UserBasedRecommender.S).toarray().ravel()
            top = rs.top5_outside_playlist(avg_sims, p, train_data, IX_tgt_playlists, UserBasedRecommender.IX_tgt_items, sim_check, secondary_sorting)
            recommendetions = np.append(recommendetions, rs.sub_format(top))

        return pd.DataFrame({'playlist_id' : IX_tgt_playlists.index.values, 'track_ids' : recommendetions})
