import numpy as np
import pandas as pd
import recsys as rs
import scipy.sparse as sps
from tqdm import tqdm

class ALSMFRecommender:

    X = None
    Y = None
    IX_items = None
    IX_playlists = None

    def __init__(self, lambda_val=0.1, alpha=40, iterations=10, rank_size=20, seed=2517):
        self.lambda_val = lambda_val
        self.alpha = alpha
        self.iterations = iterations
        self.rank_size = rank_size
        self.seed = seed
        return

    def fit(self, tracks, train_data):
        ALSMFRecommender.IX_items, _, ALSMFRecommender.IX_playlists, _ = rs.create_sparse_indexes(tracks_info=tracks, playlists=train_data)
        URM = rs.create_tgt_URM(ALSMFRecommender.IX_playlists, ALSMFRecommender.IX_items, train_data)
        URM = URM.tocsr()
        print('URM created!')
        ALSMFRecommender.X, ALSMFRecommender.Y = rs.implicit_weighted_ALS(URM, self.lambda_val, self.alpha, self.iterations, self.rank_size, self.seed)
        print("Matrix Factorization done!")
        return

    def recommend(self, tgt_tracks, tgt_playlists):
        recommendetions = np.array([])
        for playlist in tqdm(tgt_playlists['playlist_id'].values):
            p_ix = ALSMFRecommender.IX_playlists.loc[playlist]
            item_rankings = ALSMFRecommender.X[p_ix,:].dot(ALSMFRecommender.Y).toarray().ravel()
            tgt_item_rankings = item_rankings[ALSMFRecommender.IX_items.loc[tgt_tracks['track_id'].values]]
            top5_ix = np.flip(np.argsort(tgt_item_rankings)[-5:], axis=0)
            top5 = tgt_tracks['track_id'].values[top5_ix]
            recommendetions = np.append(recommendetions, rs.sub_format(top5))
        return pd.DataFrame({'playlist_id' : tgt_playlists['playlist_id'].values, 'track_ids' : recommendetions})
