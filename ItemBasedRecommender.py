import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import recsys as rs
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
#os.chdir('/Users/LucaButera/git/rschallenge')
#os.chdir('/home/giada/github/RecSys')

class ItemBasedRecommender:

    S = None
    IX_items = None
    IX_tgt_items = None
    den = None

    def __init__(self, idf=False, measure='dot', shrinkage=0, n_el_sim=20):
        self.idf = idf
        self.measure = measure
        self.shrinkage = shrinkage
        self.n_el_sim = n_el_sim
        return

    def fit(self, track_ids, train_data, tgt_tracks=None, URM=None, multiprocessing=False):
        ItemBasedRecommender.IX_items, ItemBasedRecommender.IX_tgt_items, playlists_ix, _ = rs.create_sparse_indexes(tracks_info=track_ids, playlists=train_data, tracks_reduced=tgt_tracks)
        print('Calculated Indices')

        if URM is None:
            model_URM = rs.create_tgt_URM(playlists_ix, ItemBasedRecommender.IX_items, train_data)
            model_URM = model_URM.tocsr()
            print('Model URM built')
        else:
            model_URM=URM

        if self.idf:
            model_URM = rs.ICM_idf_regularization(model_URM)
            print('Model URM regularized with IDF!')
        if ItemBasedRecommender.IX_tgt_items is not None:
            ItemBasedRecommender.S = rs.create_Smatrix(model_URM, self.n_el_sim, self.measure, self.shrinkage, ItemBasedRecommender.IX_tgt_items, ItemBasedRecommender.IX_items, multiprocessing)
        else:
            ItemBasedRecommender.S = rs.create_Smatrix(model_URM, self.n_el_sim, self.measure, self.shrinkage, multiprocessing)
            print('Similarity built')

    def recommend(self, tgt_playlists, train_data, normalize=False, H=30, sim_check=True, secondary_sorting=True, multiprocessing=False):
        _, _, IX_tgt_playlists, _ = rs.create_sparse_indexes(playlists=tgt_playlists)
        URM = rs.create_tgt_URM(IX_tgt_playlists, ItemBasedRecommender.IX_items, train_data)
        URM = URM.tocsr()
        print('URM built')

        if normalize:
            div = ItemBasedRecommender.S.sum(axis=0)
            norm_factor = div+H
        else:
            norm_factor=1

        if not multiprocessing:
            recommendetions = np.array([])
            for p in tqdm(IX_tgt_playlists.values):
                avg_sims = np.array((URM[p,:].dot(ItemBasedRecommender.S).toarray().ravel())/(norm_factor)).ravel()
                top = rs.top5_outside_playlist(avg_sims, p, train_data, IX_tgt_playlists, ItemBasedRecommender.IX_tgt_items, sim_check, secondary_sorting)
                recommendetions = np.append(recommendetions, rs.sub_format(top))
            recs = pd.DataFrame({'playlist_id' : IX_tgt_playlists.index.values, 'track_ids' : recommendetions})
        else:
            step_env = rs.RecProcessEnvironment(train_data, IX_tgt_playlists, ItemBasedRecommender.IX_tgt_items, ItemBasedRecommender.S, norm_factor, sim_check, secondary_sorting)
            sim_chunks = [i for i in range(cpu_count())]
            chunk_len = int(URM.shape[0]/cpu_count())
            chunk_flag = 0
            for i in range(cpu_count()):
                if not i+1 == cpu_count():
                    sim_chunks[i] = URM[chunk_flag:chunk_flag+chunk_len, :]
                else:
                    sim_chunks[i] = URM[chunk_flag:, :]
                chunk_flag += chunk_len
            with Pool() as pool:
                results = pool.map(step_env.step, sim_chunks)
            recs = pd.DataFrame({'playlist_id' : IX_tgt_playlists.index.values, 'track_ids' : np.concatenate(results)})
        return recs
