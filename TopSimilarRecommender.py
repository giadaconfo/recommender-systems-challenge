import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import recsys as rs
os.chdir('/Users/LucaButera/git/rschallenge')
#os.chdir('/home/giada/github/RecSys')

class TopSimilarRecommender:

    ICM = None
    S = None
    IX_items = None
    IX_tgt_items = None
    IX_tgt_playlists = None
    IX_attr = None

    def __init__(self):
        pass

    '''Requires:
        The dataset containing the tracks informations
        A list of attributes corresponding to column names in the dataset
        The dataset containing the tracks to recommend
        The minimum number of attribute occurences to keep it in the ICM
        The number of similarity elements to keep in the S matrix calculation'''
    def fit(aux, *, tracks_info=None, attributes=['artist_id', 'album', 'tags'], attributes_to_prune=None, tgt_tracks=None, n_min_attr=2, measure='dot', shrinkage=0, n_el_sim=20):
        tr_info_fixed = rs.fix_tracks_format(tracks_info)
        print('Fixed dataset')
        TopSimilarRecommender.IX_items, TopSimilarRecommender.IX_tgt_items, _, TopSimilarRecommender.IX_attr = rs.create_sparse_indexes(tracks_info=tr_info_fixed, tracks_reduced=tgt_tracks, attr_list=attributes)
        print('Calculated Indices')
        if not attributes_to_prune is None and n_min_attr >= 2:
            tr_info_fixed = delete_low_frequency_attributes(tr_info_fixed, attributes_to_prune, n_min_attr)
            print('Eliminated low frequency attributes!')
        TopSimilarRecommender.ICM = rs.create_ICM(tr_info_fixed, TopSimilarRecommender.IX_items, TopSimilarRecommender.IX_attr, attributes)
        print('ICM built')
        if TopSimilarRecommender.IX_tgt_items is not None:
            TopSimilarRecommender.S = rs.create_Smatrix(TopSimilarRecommender.ICM, n_el_sim, measure, shrinkage, TopSimilarRecommender.IX_tgt_items, TopSimilarRecommender.IX_items)
        else:
            TopSimilarRecommender.S = rs.create_Smatrix(TopSimilarRecommender.ICM, n_el_sim, measure, shrinkage)
        print('Similarity built')

    '''Requires:
        The dataset containing the playlist for which we want to make a recommendation
        The train split of the train_final dataset to use for training
        Put normalize to True to divide similarities by the item vector lenght,
        useful with many similarities and cosin measure'''
    def recommend(aux, *, tgt_playlists=None, train_playlists_tracks_pairs=None, normalize=False):
        _, _, TopSimilarRecommender.IX_tgt_playlists, _ = rs.create_sparse_indexes(playlists=tgt_playlists)
        URM = rs.create_tgt_URM(TopSimilarRecommender.IX_tgt_playlists, TopSimilarRecommender.IX_items, train_playlists_tracks_pairs)
        URM = URM.tocsr()
        print('URM built')

        recommendetions = np.array([])
        if normalize:
            norm_dividend = np.squeeze(np.asarray(TopSimilarRecommender.S.tocsc().sum(axis=0)))
            norm_dividend[norm_dividend == 0] = 1

        for p in TopSimilarRecommender.IX_tgt_playlists.values:
            if normalize:
                avg_sims = np.divide(URM[p,:].dot(TopSimilarRecommender.S).toarray(), norm_dividend).ravel()
            avg_sims = URM[p,:].dot(TopSimilarRecommender.S).toarray().ravel()
            top = rs.top5_outside_playlist(avg_sims, p, train_playlists_tracks_pairs, TopSimilarRecommender.IX_tgt_playlists, TopSimilarRecommender.IX_tgt_items)
            recommendetions = np.append(recommendetions, rs.sub_format(top))
            if (p % 1000 == 0):
                print('Recommended ' + str(p) + ' users over ' + str(TopSimilarRecommender.IX_tgt_playlists.values.shape[0]))

        return pd.DataFrame({'playlist_id' : TopSimilarRecommender.IX_tgt_playlists.index.values, 'track_ids' : recommendetions})
