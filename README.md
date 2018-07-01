
<p align="center">
<a href="https://www.polimi.it/"><img src="https://github.com/giadaconfo/data-mining-challenge/blob/master/assets/logo_polimi.png?raw=true" width="300px"></a>
</p>



# Recommender Systems Challenge

Polytechnic University of Milan

Recommender Systems  - A.Y. 2017/18

Prof. Paolo Cremonesi

Kaggle Competition: https://www.kaggle.com/c/recommender-system-2017-challenge-polimi

Team Name: Il Pollo del Profilo


## Description

We were given a dataset with data from a music streaming service and we were required to predict a list of 5 tracks for a set of playlists. The original unsplitted dataset included around 1M interactions (tracks belonging to a playlist) for 57k playlists and 100k items (tracks). The testset was composed of about 10k playlists and 32k items. The goal was to recommend a list of 5 relevant items for each playlist. MAP@5 was used for evaluation. 


## Approach

For our final solution we chose an ensemble combining three models, in order to extend their expressive power and achieve better results.
- Content Based Recommender (weight 0.4)
- Item Based Recommender (weight 0.5)
- User Based Recommender (weight 0.1)

For further information check our final [presentation](https://github.com/giadaconfo/recommender-systems-challenge/blob/master/Presentation.pdf). 
