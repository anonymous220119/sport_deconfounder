'''

Factor model validation.
- The factorization represents the lineups for each given match. 
- INPUT:
  1. dataframe of n_matches along the rows. Columns should have at least the 2 lineups and the team names. Optional: the date (to sort them).
  2. Column names: these are needed to be able to input different datasets

- STEPS (all defined in a class named 'FactorLineups')
    - Input data and extract dictionaries from api_id to id
    - Split train/test
    - Build tensor and matrix, best_formations , the inputs of the factorization algorithms
    - Run factorizations
    - Reconstruct the lineups for each match
    - Evaluate performance
    - Output results on file 
'''


#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# coding: utf8
import pandas as pd
import numpy as np
import sys
### Custom made modules
import FactorLineups as fl

'''
----------------------------------------------------------------------------------------------------------------
Parameters
----------------------------------------------------------------------------------------------------------------
'''

seed0=11  # np.random.seed(seed)
holdout_portion=0.2
competition='1729'
segments=False
weighted_by_duration=False # for game-segments only: weigh each segment by the duration --> longer segments count more
season='2014'

S=1  # number of different initializations for the factor models
S1=10  # number of random samples of 11 players per lineup; for the randomization test
alpha=0.1  # gamma shape parameter, if Gamma priors are used
verbose=True 
verbose_fact=False
F=10  # number of CV iterations
K=12 # number of latent variables
at_K2=15 # used to calculate precision@k, ncdg@k
algos=['MT','BPTF','BPMF'] # algorithms used to perform inference
recon_flag=True # whether to use the reconstructed causes or the plain confounder in k-dim
flag_mask=True # whether to multiply the reconstructed lineup by a binary mask for the team formation
stratified=True # whether to use a stratified split: each pair of team defines a match 
with_repetitions=False # if False a pair of teams (a,b) will not appear in both train and test sets, either on the train OR the test set (XOR)
 
'''
----------------------------------------------------------------------------------------------------------------
INPUT parameters from command line
----------------------------------------------------------------------------------------------------------------
'''
c=0
for arg in sys.argv:
    print arg,
    if c==1: competition=str(arg)
    elif c==2: K=int(arg)
    elif c==3: season=str(arg)
    elif c==4: segments=bool(arg)
    elif c==5: weighted_by_duration=bool(arg)
    elif c==6: F=int(arg)
    elif c==7: S=int(arg)
    elif c==8: seed0=int(arg)
    c+=1;
print sys.argv[1:];


if segments==False:
    outfile='../data/output/factor_model_evaluation_'+str(competition)+'_starting_lineup_'+season
    try:
        df=pd.read_csv('../data/output/'+competition+'_starting_lineup.csv')
    except:df=pd.read_csv('../data/output/'+competition+'_starting_lineup_'+season+'.csv')
else:
    outfile='../data/output/factor_model_evaluation_'+str(competition)+'_segments_'+season
    if weighted_by_duration==True:outfile=outfile+'_weighted'
    df=pd.read_csv('../data/output/'+competition+'_segments_'+season+'.csv')

print('Output file:', outfile)

'''
Define dataframe columns names
'''
try:df=df.sort_values(by=['date']) # optional
except:1

cols=df.columns.values

cols_home=[c for c in cols if 'home_player_' in c and 'name' not in c]
cols_away=[c for c in cols if 'away_player_' in c and 'name' not in c]
cols_home_team='home_team_api_id'
cols_away_team='away_team_api_id'
assert len(cols_home)==len(cols_away)
at_K=len(cols_home)

FL=fl.FactorLineups(df,cols_home=cols_home,cols_away=cols_away,cols_home_team=cols_home_team,cols_away_team=cols_away_team,seed_cv=seed0,K=K,segments=segments,mask_flag=flag_mask,verbose=verbose,verbose_fact=verbose_fact,S=S)

df_results=FL.cross_validation(df,holdout_portion=holdout_portion,F=F,algos=algos,recon_flag=recon_flag,at_K=at_K,at_K2=at_K2,outfile=outfile,stratified=stratified,with_repetitions=with_repetitions,weighted_by_duration=weighted_by_duration)

