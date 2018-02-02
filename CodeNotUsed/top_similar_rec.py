import numpy as np
import pandas as pd
import os.path
from scipy import sparse as sps
import sys
os.chdir('/Users/LucaButera/git/rschallenge')
#os.chdir('/home/giada/github/RecSys')

def recommend(pls):
    recommendetions = np.array([])
    #norm_dividend = np.squeeze(np.asarray(S.tocsc().sum(axis=0)))
    #norm_dividend[norm_dividend == 0] = 1

    for p in pls:
        #avg_sims = np.divide(P_T[p,:].dot(S).toarray(), norm_dividend).ravel()
        avg_sims = P_T[p,:].dot(S).toarray().ravel()
        top = top5_outside_playlist(avg_sims, p)
        recommendetions = np.append(recommendetions, sub_format(top))
        print(p)

    return pd.DataFrame(data=np.array([INDEX_pl.index.values, recommendetions]).transpose() , index=range(pls.size), columns=['playlist_id', 'track_ids'])

def top5_outside_playlist(ratings, p_id):
    tgt_in_playlist = np.intersect1d(train_final[train_final['playlist_id'] == INDEX_pl.index.values[p_id]]['track_id'].values, ICM_tgt_items.index.values, assume_unique=True)
    ratings[ICM_tgt_items.loc[tgt_in_playlist]['track_id'].values] = 0 #line to change

    if(np.count_nonzero(ratings) < 5): sys.exit('Not enough similarity')

    top5_ind = np.flip(np.argsort(ratings)[-5:], axis=0) #Contains the index of the recommended songs
    return ICM_tgt_items.index.values[top5_ind]

def sub_format(l):
    res = " ".join(np.array_str(l).split())[1:-1]
    return res

ICM_items = pd.read_csv('BuiltStructures/ICM_items.csv', index_col=0, header=None, names=['track_id'])
ICM_tgt_items = pd.read_csv('BuiltStructures/ICM_tgt_items.csv', index_col=0, header=None, names=['track_id'])
INDEX_pl = pd.read_csv('BuiltStructures/INDEX_pl.csv', index_col=0, header=None, names=['playlist_id'])
S = sps.load_npz('BuiltStructures/RecommendableSimilarityMatrix.npz').tocsc()
P_T = sps.load_npz('BuiltStructures/PL_TR_MAT.npz').tocsr()
train_final = pd.read_csv('Data/train_final.csv','\t')

res = recommend(INDEX_pl['playlist_id'].values)

res.to_csv('Submissions/top_similar_submission.csv', index=False)
print(res)
