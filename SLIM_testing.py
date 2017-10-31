import numpy as np
import pandas as pd
import recsys as rs
import os.path
import SLIM_BPR_PY as SLIMBPR
os.chdir('/Users/LucaButera/git/rschallenge')

dataset = pd.read_csv('Data/train_final.csv','\t')
tr_info = pd.read_csv('Data/tracks_final.csv','\t')

IX_items, _, IX_tgt_playlists, _ = rs.create_sparse_indexes(tracks_info=tr_info, playlists=dataset)

URM = rs.create_tgt_URM(IX_tgt_playlists, IX_items, dataset)
URM = URM.tocsr()

recommender = SLIMBPR.SLIM_BPR_Python(URM, sparse_weights=True)
recommender.fit(batch_size = 1, topK=20)

recommendetions = np.array([])
for p in IX_tgt_playlists.values:
            avg_sims = URM[p,:].dot(recommender.S).toarray().ravel()
            top = rs.top5_outside_playlist(avg_sims, p, train, IX_tgt_playlists, IX_tgt_items)
            recommendetions = np.append(recommendetions, rs.sub_format(top))
            if (p % 1000 == 0):
                print('Recommended ' + str(p) + ' users over ' + str(IX_tgt_playlists.values.shape[0]))
print(recommendetions.shape[0])
print(tgt_playlists['playlist_id'].shape[0])
print(IX_tgt_playlists.shape[0])
recs = pd.DataFrame({'playlist_id' : IX_tgt_playlists.index.values, 'track_ids' : recommendetions})

recs.to_csv('Submissions/slim_recs.csv', index=False)
