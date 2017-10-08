import numpy as np
import pandas as pd
import os.path
from scipy.sparse import *
os.chdir('/Users/LucaButera/git/rschallenge') #modify this according to your environment

ICM_items = pd.read_csv('BuiltStructures/ICM_items.csv')
ICM_tgt_items = pd.read_csv('BuiltStructures/ICM_tgt_items.csv')
INDEX_pl = pd.read_csv('BuiltStructures/INDEX_pl.csv')
S = load_npz('BuiltStructures/RecommendableSimilarityMatrix.npz')
P_T = load_npz('BuiltStructures/PL_TR_MAT.npz')

pl = pd.read_csv('Data/train_final.csv','\t')
rec_tr = pd.read_csv('Data/target_tracks.csv','\t')
rec_pl = pd.read_csv('Data/target_playlists.csv','\t')

for p in INDEX_pl.index:
