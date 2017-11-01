import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import recsys as rs
os.chdir('/Users/LucaButera/git/rschallenge')

class ItemBasedRecommender:

    S = None
    IX_items = None
    IX_tgt_items = None

    def __init__(self, idf=False, measure='dot', shrinkage=0, n_el_sim=20):
        self.idf = idf
        self.measure = measure
        self.shrinkage = shrinkage
        self.n_el_sim = n_el_sim
        return


    def fit(self, track_ids, train_data, tgt_tracks=None):
        ItemBasedRecommender.IX_items, ItemBasedRecommender.IX_tgt_items, playlists_ix, _ = rs.create_sparse_indexes(tracks_info=track_ids, playlists=train_data, tracks_reduced=tgt_tracks)
        print('Calculated Indices')

        model_URM = rs.create_tgt_URM(playlists_ix, ItemBasedRecommender.IX_items, train_data)
        model_URM = model_URM.tocsr()
        print('Model URM built')

        if self.idf:
            model_URM = rs.ICM_idf_regularization(model_URM)
            print('Model URM regularized with IDF!')
        if ItemBasedRecommender.IX_tgt_items is not None:
            ItemBasedRecommender.S = rs.create_Smatrix(model_URM, self.n_el_sim, self.measure, self.shrinkage, ItemBasedRecommender.IX_tgt_items, ItemBasedRecommender.IX_items)
        else:
            ItemBasedRecommender.S = rs.create_Smatrix(model_URM, self.n_el_sim, self.measure, self.shrinkage)
        print('Similarity built')

    def recommend(self, tgt_playlists, train_data, sim_check=True, secondary_sorting=True):
        _, _, IX_tgt_playlists, _ = rs.create_sparse_indexes(playlists=tgt_playlists)
        URM = rs.create_tgt_URM(IX_tgt_playlists, ItemBasedRecommender.IX_items, train_data)
        URM = URM.tocsr()
        print('URM built')

        recommendetions = np.array([])
        for p in IX_tgt_playlists.values:
            avg_sims = URM[p,:].dot(ItemBasedRecommender.S).toarray().ravel()
            top = rs.top5_outside_playlist(avg_sims, p, train_data, IX_tgt_playlists, ItemBasedRecommender.IX_tgt_items, sim_check, secondary_sorting)
            recommendetions = np.append(recommendetions, rs.sub_format(top))
            if (p % 1000 == 0):
                print('Recommended ' + str(p) + ' users over ' + str(IX_tgt_playlists.values.shape[0]))
        return pd.DataFrame({'playlist_id' : IX_tgt_playlists.index.values, 'track_ids' : recommendetions})
