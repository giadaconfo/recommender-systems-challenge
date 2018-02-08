import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import recsys as rs
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
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

    def fit(self, track_ids, train_data, tgt_playlists, multiprocessing=False):
        UserBasedRecommender.IX_items, UserBasedRecommender.IX_tgt_items, UserBasedRecommender.IX_playlists, _ = rs.create_sparse_indexes(tracks_info=track_ids, playlists=train_data)
        _, _, UserBasedRecommender.IX_tgt_playlists, _ = rs.create_sparse_indexes(playlists=tgt_playlists)
        print('Calculated Indices')

        model_URM = rs.create_tgt_URM(UserBasedRecommender.IX_playlists, UserBasedRecommender.IX_items, train_data)
        model_URM = model_URM.tocsr()
        print(model_URM.shape)
        print('Model URM built')

        if self.idf:
            model_URM = rs.ICM_idf_regularization(model_URM)
            print('Model URM regularized with IDF!')

        UserBasedRecommender.S = rs.create_Smatrix(model_URM.T, self.n_el_sim, self.measure, self.shrinkage, UserBasedRecommender.IX_tgt_playlists, UserBasedRecommender.IX_playlists, multiprocessing)
        print('Similarity built')

    def recommend(self, tgt_items, train_data, normalize=False, H=20, sim_check=True, secondary_sorting=True, multiprocessing=False):
        _, UserBasedRecommender.IX_tgt_items, _, _ = rs.create_sparse_indexes(tracks_reduced=tgt_items)
        URM = rs.create_UBR_URM(UserBasedRecommender.IX_playlists, UserBasedRecommender.IX_tgt_items, train_data)
        URM = URM.tocsr()
        print('URM built')

        UserBasedRecommender.S = UserBasedRecommender.S.T.tocsr()
        if normalize:
            div = UserBasedRecommender.S.sum(axis=0)
            norm_factor = div+H
        else:
            norm_factor=1

        if not multiprocessing:
            recommendetions = np.array([])
            for p in tqdm(UserBasedRecommender.IX_tgt_playlists.values):
                avg_sims = (UserBasedRecommender.S[p,:].multiply(1/(norm_factor)).dot(URM).toarray().ravel())
                top = rs.top5_outside_playlist(avg_sims, p, train_data, UserBasedRecommender.IX_tgt_playlists, UserBasedRecommender.IX_tgt_items, sim_check, secondary_sorting)
                recommendetions = np.append(recommendetions, rs.sub_format(top))
            recs = pd.DataFrame({'playlist_id' : UserBasedRecommender.IX_tgt_playlists.index.values, 'track_ids' : recommendetions})
        else:
            step_env = rs.RecProcessEnvironment(train_data, UserBasedRecommender.IX_tgt_playlists, UserBasedRecommender.IX_tgt_items, URM, norm_factor, sim_check, secondary_sorting, userbased=True)
            sim_chunks = [i for i in range(cpu_count())]
            chunk_len = int(UserBasedRecommender.S.shape[0]/cpu_count())
            chunk_flag = 0
            for i in range(cpu_count()):
                if not i+1 == cpu_count():
                    sim_chunks[i] = UserBasedRecommender.S[chunk_flag:chunk_flag+chunk_len, :]
                else:
                    sim_chunks[i] = UserBasedRecommender.S[chunk_flag:, :]
                chunk_flag += chunk_len
            with Pool() as pool:
                results = pool.map(step_env.step, sim_chunks)
            recs = pd.DataFrame({'playlist_id' : UserBasedRecommender.IX_tgt_playlists.index.values, 'track_ids' : np.concatenate(results)})
        return recs
