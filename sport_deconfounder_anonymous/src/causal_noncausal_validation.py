import numpy as np
import pandas as pd
import sys
from argparse import ArgumentParser

import FactorLineups as fl
import SportDeconfounder as sd


p=sd.argparse_sd()
args = p.parse_args() # INPUT parameters from command line
print(args.usebestk)
if args.usebestk==True:
	infile_best_K='../data/output/best_k_factor_model.dat'
	args.K=sd.import_best_k(infile_best_K, league=args.league,season=args.season,algo=args.algo,default_k=args.K)

if args.segments==False:
	outfile='../data/output/causal_noncausal_starting_lineup'
	try:
		df=pd.read_csv('../data/output/'+args.league+'_starting_lineup.csv')
	except:df=pd.read_csv('../data/output/'+args.league+'_starting_lineup_'+args.season+'.csv')
else:
	outfile='../data/output/causal_noncausal_segments'
	if args.weighted_by_duration==True:outfile=outfile+'_weighted'
	df=pd.read_csv('../data/output/'+args.league+'_segments_'+args.season+'.csv')

print('Output file:', outfile)
print(args.league,args.season,args.split_type,args.K)

'''
Define dataframe columns names
'''
try:df=df.sort_values(by=[args.col_date]) 
except:print('Warning: DataFrame not sorted by date')

cols_home_score='home_team_goal'
cols_away_score='away_team_goal'

cols=df.columns.values

cols_home=[c for c in cols if 'home_player_' in c and 'name' not in c]
cols_away=[c for c in cols if 'away_player_' in c and 'name' not in c]
cols_home_team='home_team_api_id'
cols_away_team='away_team_api_id'
assert len(cols_home)==len(cols_away)

FL=fl.FactorLineups(df,cols_home=cols_home,cols_away=cols_away,cols_home_team=cols_home_team,cols_away_team=cols_away_team,seed_cv=args.seed0,K=args.K,segments=args.segments,mask_flag=args.mask_flag,verbose=args.verbose,verbose_fact=args.verbose_fact,S=args.S)
SD=sd.SportDeconfounder(df,FL,verbose=args.verbose_om,Z=None,A_recon=None,algo=args.algo,weighted_by_duration=args.weighted_by_duration) # produces Z and A_recon

df_results=SD.causal_noncausal_cross_validation(df,FL,cols_home_score=cols_home_score,cols_away_score=cols_away_score,outfile=outfile,solver=args.solver,recon_flag=args.recon_flag,alphas=None,test=args.test,outcome_cat=args.outcome_cat,weighted_by_duration=args.weighted_by_duration,algos=[args.algo],match_days=args.match_days,split_type=args.split_type,seed0=args.seed0,folds=args.F,results_name=[args.league,args.season,args.K,args.match_days,args.split_type],epsilon=args.epsilon,min_caps=args.min_caps,normalize=args.normalize,n_iter_alpha=args.n_iter_alpha)

