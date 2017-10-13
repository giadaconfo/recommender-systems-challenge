import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import sys
import recsys as rs
os.chdir('/Users/LucaButera/git/rschallenge')
#os.chdir('/home/giada/github/RecSys')

class TopSimilarRecommender:
    def __init__(self):
        self.ICM = None
        self.S = None
        self.IX_items = None
        self.IX_tgt_items = None
        self.IX_tgt_playlists = None
        self.IX_attr = None

    '''Requires:
        The dataset containing the tracks informations
        A list of attributes corresponding to column names in the dataser
        The minimum number of attribute occurences to keep it in the ICM
        The number of similarity elements to keep in the S matrix calculation'''
    def fit(tracks_info, attributes, n_min_attr, measure, n_el_sim, Smat=None):
        tr_info_fixed = rs.fix_tracks_format(tracks_info)
        self.IX_items, self.IX_tgt_items, self.IX_tgt_playlists, self.IX_attr = rs.create_sparse_indexes(tr_info_fixed, attr_list=attributes)
        self.ICM = rs.create_ICM(tr_info_fixed, self.IX_items, self.IX_attr, n_min_attr)
        self.S = rs.create_Smatrix(self.ICM, n_el_sim, measure)

    '''Requires:
        The dataset containing the playlist for which we want to make a recommendation
        The train split of the train_final dataset to use for training'''
    def recommend(tgt_playlists, train_playlists_tracks_pairs):
        URM = rs.create_tgt_URM(self.IX_tgt_playlists, self.IX_items, train_playlists_tracks_pairs)

        tgt_playlists_ix = self.IX_tgt_playlists.loc[tgt_playlists['playlist_id'].values]

        recommendetions = np.array([])
        #norm_dividend = np.squeeze(np.asarray(S.tocsc().sum(axis=0)))
        #norm_dividend[norm_dividend == 0] = 1

        for p in tgt_playlists_ix:
            #avg_sims = np.divide(P_T[p,:].dot(S).toarray(), norm_dividend).ravel()
            avg_sims = URM[p,:].dot(self.S).toarray().ravel()
            top = rs.top5_outside_playlist(avg_sims, p)
            recommendetions = np.append(recommendetions, rs.sub_format(top))
            print(p)

        return pd.DataFrame(data=np.array([tgt_playlists['playlist_id'].values, recommendetions]).transpose() , index=range(tgt_playlists_ix.size), columns=['playlist_id', 'track_ids'])
