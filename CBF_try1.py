import pandas as pd
import numpy as np
import os.path
pd.set_option('display.max_columns',500)
os.chdir('/Users/LucaButera/git/rschallenge')

train = pd.read_csv('Data/train_final.csv','\t')
tr_info = pd.read_csv('Data/tracks_final.csv','\t')
pl_info = pd.read_csv('Data/playlists_final.csv','\t')
tgt_pl = pd.read_csv('Data/target_playlists.csv','\t')
tgt_tr = pd.read_csv('Data/target_tracks.csv','\t')

def splitnflat(v):
    res = []
    for x in np.nditer(v, flags=['refs_ok']):
        new = [int(j[0]) for j in [x.item(0)[1:-1].split(',')] if j[0] != 'None' and j[0] != '']
        res = res + new
    return np.unique(np.array(res))

items = tr_info['track_id'].values
albums = splitnflat(tr_info['album'].values)
tags = splitnflat(tr_info['tags'].values)
attr = np.append(albums, tags)

ICM = pd.DataFrame(index=attr, columns=items)
print(ICM.shape)
