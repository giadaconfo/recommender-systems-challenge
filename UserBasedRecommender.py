import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import recsys as rs
from tqdm import tqdm
#os.chdir('/Users/LucaButera/git/rschallenge')
#os.chdir('/home/giada/github/RecSys')

class UserBasedRecommender:

    S = None
    IX_users =None
    IX_items = None
    IX_tgt_items = None
    IX_playlists = None
    IX_tgt_playlists = None

    def __init__(self, idf=False, measure='cos', shrinkage=10, n_el_sim=20):
        self.idf = idf
        self.measure = measure
        self.shrinkage = shrinkage
        self.n_el_sim = n_el_sim
        return

    '''
    @saed_similarity: matrix to import
    @save_sim: set true to save the matrix that will be created
    '''
    def fit(self, track_ids, train_data, tgt_playlists, saved_similarity=None, save_sim=False):
        UserBasedRecommender.IX_items, _, UserBasedRecommender.IX_playlists, _ = rs.create_sparse_indexes(tracks_info=track_ids, playlists=train_data)
        _, _, UserBasedRecommender.IX_tgt_playlists, _ = rs.create_sparse_indexes(playlists=tgt_playlists)
        print('Calculated Indices')

        model_URM = rs.create_tgt_URM(UserBasedRecommender.IX_playlists, UserBasedRecommender.IX_items, train_data)
        model_URM = model_URM.tocsr()
        print(model_URM.shape)
        print('Model URM built')

        if self.idf:
            model_URM = rs.ICM_idf_regularization(model_URM)
            print('Model URM regularized with IDF!')

        if saved_similarity is None:
            UserBasedRecommender.S = rs.create_Smatrix(model_URM.T, self.n_el_sim, self.measure, self.shrinkage, UserBasedRecommender.IX_tgt_playlists, UserBasedRecommender.IX_playlists)
            print('Similarity built')
            if save_sim:
                sps.save_npz('BuiltStructures/ubr_sim_65el_impcos.npz', UserBasedRecommender.S)
        else:
            UserBasedRecommender.S = sps.load_npz(saved_similarity).tocsr()
    def recommend(self, tgt_items, train_data, sim_check=True, secondary_sorting=True):
        _, UserBasedRecommender.IX_tgt_items, _, _ = rs.create_sparse_indexes(tracks_reduced=tgt_items)
        URM = rs.create_UBR_URM(UserBasedRecommender.IX_playlists, UserBasedRecommender.IX_tgt_items, train_data)
        URM = URM.tocsr()
        print('URM built')

        recommendetions = np.array([])
        UserBasedRecommender.S = UserBasedRecommender.S.T.tocsr()
        for p in UserBasedRecommender.IX_tgt_playlists.values:
            avg_sims = UserBasedRecommender.S[p,:].dot(URM).toarray().ravel()
            top = rs.top5_outside_playlist(avg_sims, p, train_data, UserBasedRecommender.IX_tgt_playlists, UserBasedRecommender.IX_tgt_items, sim_check, secondary_sorting)
            recommendetions = np.append(recommendetions, rs.sub_format(top))

        return pd.DataFrame({'playlist_id' : UserBasedRecommender.IX_tgt_playlists.index.values, 'track_ids' : recommendetions})
