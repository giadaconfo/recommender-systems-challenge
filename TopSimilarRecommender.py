import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import recsys as rs
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
#os.chdir('/Users/LucaButera/git/rschallenge')
#os.chdir('/home/giada/github/RecSys')

class TopSimilarRecommender:

    ICM = None
    S = None
    IX_items = None
    IX_tgt_items = None
    IX_tgt_playlists = None
    IX_attr = None
    den = None

    def __init__(self, attributes=['artist_id', 'album', 'tags'], attributes_to_prune=None, n_min_attr=2, idf=False, measure='dot', shrinkage=0, n_el_sim=20):
        self.attributes = attributes
        self.attributes_to_prune = attributes_to_prune
        self.n_min_attr = n_min_attr
        self.idf = idf
        self.measure = measure
        self.shrinkage = shrinkage
        self.n_el_sim = n_el_sim
        return

    '''
    @saved_similarity: matrix to import
    @save_sim: set true to save the matrix that will be created
    '''
    def fit(self, tracks_info, tgt_tracks=None, saved_similarity=None, save_sim=False, multiprocessing=False):
        tr_info_fixed = rs.fix_tracks_format(tracks_info)
        print('Fixed dataset')
        TopSimilarRecommender.IX_items, TopSimilarRecommender.IX_tgt_items, _, TopSimilarRecommender.IX_attr = rs.create_sparse_indexes(tracks_info=tr_info_fixed, tracks_reduced=tgt_tracks, attr_list=self.attributes)
        print('Calculated Indices')
        if not self.attributes_to_prune is None and self.n_min_attr >= 2:
            tr_info_fixed = rs.delete_low_frequency_attributes(tr_info_fixed, self.attributes_to_prune, self.n_min_attr)
            print('Eliminated low frequency attributes!')
        TopSimilarRecommender.ICM = rs.create_ICM(tr_info_fixed, TopSimilarRecommender.IX_items, TopSimilarRecommender.IX_attr, self.attributes)
        print('ICM built')
        if self.idf:
            TopSimilarRecommender.ICM = rs.ICM_idf_regularization(TopSimilarRecommender.ICM)
            print('ICM regularized with IDF!')
        if saved_similarity is None:
            if TopSimilarRecommender.IX_tgt_items is not None:
                TopSimilarRecommender.S = rs.create_Smatrix(TopSimilarRecommender.ICM, self.n_el_sim, self.measure, self.shrinkage, TopSimilarRecommender.IX_tgt_items, TopSimilarRecommender.IX_items, multiprocessing)
            else:
                TopSimilarRecommender.S = rs.create_Smatrix(TopSimilarRecommender.ICM, self.n_el_sim, self.measure, self.shrinkage, multiprocessing)
            print('Similarity built')
            if save_sim:
                sps.save_npz('BuiltStructures/tsr_sim_65el_idfTrue_artist_album_playcount.npz', TopSimilarRecommender.S)
        else:
            TopSimilarRecommender.S = sps.load_npz(saved_similarity).tocsr()




    def recommend(self, tgt_playlists, train_playlists_tracks_pairs, normalize=False, H=30, sim_check=True, secondary_sorting=True, multiprocessing=False):
        _, _, TopSimilarRecommender.IX_tgt_playlists, _ = rs.create_sparse_indexes(playlists=tgt_playlists)
        URM = rs.create_tgt_URM(TopSimilarRecommender.IX_tgt_playlists, TopSimilarRecommender.IX_items, train_playlists_tracks_pairs)
        URM = URM.tocsr()
        print('URM built')

        if normalize:
            div = TopSimilarRecommender.S.sum(axis=0)
            norm_factor = div+H
        else:
            norm_factor=1

        if not multiprocessing:
            recommendetions = np.array([])
            for p in tqdm(TopSimilarRecommender.IX_tgt_playlists.values):
                avg_sims = (URM[p,:].dot(TopSimilarRecommender.S).toarray().ravel())/(norm_factor)
                top = rs.top5_outside_playlist(avg_sims, p, train_playlists_tracks_pairs, TopSimilarRecommender.IX_tgt_playlists, TopSimilarRecommender.IX_tgt_items, sim_check, secondary_sorting)
                recommendetions = np.append(recommendetions, rs.sub_format(top))
            recs = pd.DataFrame({'playlist_id' : TopSimilarRecommender.IX_tgt_playlists.index.values, 'track_ids' : recommendetions})
        else:
            step_env = rs.RecProcessEnvironment(train_playlists_tracks_pairs, TopSimilarRecommender.IX_tgt_playlists, TopSimilarRecommender.IX_tgt_items, TopSimilarRecommender.S, norm_factor, sim_check, secondary_sorting)
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
