import pandas as pd
import numpy as np
import os.path
os.chdir('/home/giada/github/RecSys') #modify this according to your environment

tr_info = pd.read_csv('Data/tracks_final.csv','\t')

def fix_artists(val):
    return int(val[1:-1]) if val != '[None]' and val != '[]' else 'NaN'

def fix_tags(val):
    return pd.Series(val[1:-1].split(', '), dtype='int') if val != '[None]' and val != '[]' else pd.Series(['NaN']*5)

tr_info['album'] = tr_info['album'].apply(fix_artists)
tr_info[['tag1','tag2','tag3','tag4','tag5']] = tr_info['tags'].apply(fix_tags)
tr_info.drop(labels='tags', axis=1, inplace=True)

tr_info.to_csv('Data/fixed_tracks_final.csv')
