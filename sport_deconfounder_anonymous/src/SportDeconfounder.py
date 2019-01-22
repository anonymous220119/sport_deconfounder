import numpy as np
import pandas as pd
import glob
from argparse import ArgumentParser

import FactorLineups as fl
import OutcomeModel as om

class SportDeconfounder :
    def __init__(self,df,FL,verbose=False,Z=None,A_recon=None,fact_obj=None,algo='BPTF',weighted_by_duration=False):  

        self.verbose=verbose
        self.Z=Z
        self.A_recon=A_recon
        self.fact_obj=fact_obj
        self.algo=algo

        if self.Z is None:
            assert self.A_recon is None
            self.run_factorization(df,FL,algos=[algo],weighted_by_duration=weighted_by_duration)


    def run_factorization(self,df,FL,algos=['BPTF'],weighted_by_duration=False):

        self.Z,self.A_recon,self.fact_obj=FL.run_factorization(df,algos=algos,weighted_by_duration=weighted_by_duration)


    def run_outcome_model(self,df,OM,FL,solver='auto',alphas=None,test=False,seed=10,holdout_portion_alpha_ridge=0.2):
        '''
        X_aug,regr are both dict with keys 'causal', 'non_causal'
        '''
        cf_dict=(FL.fit(df,self.Z,self.A_recon,recon_flag=OM.recon_flag))[self.algo] # n_match rows, n_cols could be either K or n_players based on recon_flag
        confounder=cf_dict['home']-cf_dict['away']
        alpha_ridge=OM.estimate_alpha_ridge(confounder,solver=solver,alphas=alphas,test=test,holdout_portion=holdout_portion_alpha_ridge)
        X_aug,regr=om.causal_noncausal_regr(OM.X,OM.Y,confounder,alpha_ridge,normalize=OM.normalize,seed=seed,solver=solver)
        
        return X_aug,regr,alpha_ridge


    def causal_noncausal_cross_validation(self,df,FL,cols_home_score='home_team_goal',cols_away_score='away_team_goal',outfile=None,solver='auto',recon_flag=True,alphas=None,test=False,outcome_cat=[-3,0,3],weighted_by_duration=False,algos=['BPTF'],match_days=8,split_type=2,seed0=10,folds=1,results_name=['10257','2016','5','8','2'],vad_portion=0.1,normalize=True,n_iter_alpha=10,n_iter=10,min_caps=0,epsilon=0.1,weights=None):

        results=[]

        if split_type==2:F=folds # random split
        else: F=1 # split is fixed

        for f in range(F):
            print('f:',f)

            X_aug_train={};X_aug_test={}

            train_df,test_df=train_test_split(df,match_days=match_days,n_teams=FL.n_teams,split_type=split_type,seed0=seed0+f)

            OM_train=om.OutcomeModel(train_df,FL,cols_home_score=cols_home_score,cols_away_score=cols_away_score,seed_cv=seed0,vad_portion=vad_portion,recon_flag=recon_flag,normalize=normalize,verbose=self.verbose,n_iter=n_iter,n_iter_cv=n_iter_alpha,min_caps=min_caps,epsilon=epsilon,weights=weights)
            OM_test=om.OutcomeModel(test_df,FL,cols_home_score=cols_home_score,cols_away_score=cols_away_score,recon_flag=recon_flag,normalize=normalize,verbose=self.verbose,min_caps=min_caps,epsilon=epsilon,weights=weights)

            X_aug_train,regr,alpha_ridge=self.run_outcome_model(train_df,OM_train,FL,solver=solver,alphas=alphas,test=test,seed=seed0,holdout_portion_alpha_ridge=OM_train.vad_portion)

            # confounder_train=FL.fit(train_df,self.Z,self.A_recon,recon_flag=recon_flag)
            cf_dict=(FL.fit(test_df,self.Z,self.A_recon,recon_flag=OM_test.recon_flag))[self.algo]
            confounder_test=cf_dict['home']-cf_dict['away']
            X_aug_test['non_causal'] =np.column_stack([OM_test.causes, np.zeros(confounder_test.shape)]) 
            X_aug_test['causal'] =np.column_stack([OM_test.causes, confounder_test])

            intercept=np.mean(OM_train.Y,axis=0) # baseline

            '''
            Evaluate performance
            '''
            results_f_values=performance_evaluation(regr,Y_train=OM_train.Y,Y_test=OM_test.Y,X_train=X_aug_train,X_test=X_aug_test,Y_train_categorical=OM_train.Y_categorical,Y_test_categorical=OM_test.Y_categorical,intercept=intercept,epsilon=epsilon,outcome_cat=outcome_cat,verbose=self.verbose)
            
            for a in ['non_causal','causal']:
                results_f=list(results_name)
                results_f.extend([f,a])
                results_f.extend(list(np.concatenate([list(results_f_values[i][a]) for i in range(len(results_f_values))] )))

                results.append(results_f)
        
                if self.verbose==True:print f,a,results_f

        df_res=output_results_onfile(results,outfile=outfile,cols_res=result_labels())

        return df_res


def train_test_split(df,match_days=8,n_teams=1,split_type=2,seed0=-1):
    ''' 
    To compare causaul vs non_causal
    Note: DataFrame df has to be already sorted by date
    '''
    n_test=match_days*n_teams/2
    if split_type==0: # from the beginning of the season
        test_df=df.iloc[:n_test]
        train_df=df.iloc[n_test:]
    elif split_type==1: # at the end of the season
        test_df=df.iloc[-n_test:]
        train_df=df.iloc[:-n_test]
    elif split_type==2:  # random order
        if seed0==-1:seed=np.random.randint(123456)
        else:seed=np.copy(seed0)
        prng=np.random.RandomState(seed)
        permuted=prng.permutation(np.arange(len(df)))
        test_df=df.iloc[permuted[:n_test]]
        train_df=df.iloc[permuted[n_test:]]

    return train_df,test_df


def output_results_onfile(results,outfile,cols_res=None):
    
    assert cols_res is not None
    df=pd.DataFrame(results,columns=cols_res)
    if outfile is not None:
        files_present = glob.glob(outfile+'.csv')
        if not files_present:
            df.to_csv(outfile+'.csv', index=False, header=True)
        else:
            print('File ',outfile,'already exists')
            df.to_csv(outfile+'.csv', index=False, header=False, mode='a')
        print('Outfile:',outfile+'.csv')
    return df


def result_labels():
    labels=['league','season','K','match_days','split_type','fold','model_type']
    labels.extend(['mae_tr','mae_te','mae_tr_bs','mae_te_bs'])
    labels.extend(['mse_tr','mse_te','mse_tr_bs','mse_te_bs'])
    labels.extend(['acc_tr','acc_te','acc_tr_bs','acc_te_bs'])

    return labels


def performance_evaluation(regr,X_train=None,X_test=None,Y_train=None,Y_test=None,Y_train_categorical=None,Y_test_categorical=None,intercept=None,epsilon=0.1,outcome_cat=[-3,0,3],verbose=False):
    mae={};mse={};acc={}
    if intercept is None: intercept=np.mean(Y_train,axis=0) # baseline

    for model in regr: # models are 'causal' 'non_causal'

        Y_pred_train = regr[model].predict(X_train[model])
        Y_pred_test = regr[model].predict(X_test[model])

        mae[model]=om.mae_output(Y_train,Y_test,Y_pred_train,Y_pred_test,intercept)
        mse[model]=om.mse_output(Y_train,Y_test,Y_pred_train,Y_pred_test,intercept)
        if Y_train_categorical is not None:
            Y_pred_train_sharp=om.sharp_predict(Y_pred_train,epsilon=epsilon,outcome=outcome_cat)
            y_pred_test_sharp=om.sharp_predict(Y_pred_test,epsilon=epsilon,outcome=outcome_cat)
            acc[model]=om.accuracy_output(Y_train_categorical,Y_test_categorical,Y_pred_train_sharp,y_pred_test_sharp,verbose=verbose)

    return (mae,mse,acc)

def import_best_k(infile='../data/output/best_k_factor_model.dat', league=10257,season=2016,algo='BPTF',default_k=1):
    try:
        df=pd.read_csv(infile)
        best_k=df[(df.algo==algo) & (df.league==int(league)) & (df.season==int(season))].best_K.values[0]
    except: best_k=default_k
    return best_k

def extract_coeff_linear_regression(lm,labels=None,rounding=4, coef_name='beta', player_name="player_name"):
    '''
    Input: regr is a dict with keys 'causal', 'non_causal'; values are Ridge regression objects
    '''
    df = pd.DataFrame()
    
    params=[]
    for i in lm:
        params.append(np.append(lm[i].intercept_,lm[i].coef_))
        params[-1] = np.round(params[-1],rounding)
        df[coef_name+"_"+i] = params[-1]

    if labels is not None:
        df[player_name] = labels

    return df
'''
------------------------------------------------------------------------------------------------------------------------
Command line parser
------------------------------------------------------------------------------------------------------------------------
'''

def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]

def add_boolean_argument(parser, name, default=False):                                                                                               
    """Add a boolean argument to an ArgumentParser instance."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('--no' + name, dest=name, action='store_false')

def argparse_sd():
    p = ArgumentParser()

    p.add_argument('-l', '--league', type=str, default='10257')
    p.add_argument('-s', '--season', type=str, default='2016')
    p.add_argument('-a', '--algo', type=str, default='BPTF')
    p.add_argument('-col_date', '--col_date', type=str, default='date')
    p.add_argument('-split', '--split_type', type=int, default=0)
    p.add_argument('-k', '--K', type=int, default=5)
    p.add_argument('-g', '--match_days', type=int, default=8)
    p.add_argument('-f', '--F', type=int, default=100)
    p.add_argument('-outcome_cat', '--outcome_cat', type=list, default=[-3,0,3])
    p.add_argument('-solver', '--solver', type=str, default='auto')
    p.add_argument('-S', '--S', type=int, default=10)
    p.add_argument('-seed', '--seed0', type=int,default=10)
    p.add_argument('-caps', '--min_caps', type=int,default=0)
    p.add_argument('-n_iter_alpha', '--n_iter_alpha', type=int,default=10)
    p.add_argument('-e', '--epsilon', type=float,default=0.1)

    '''
    Boolean arguments have a special treatment, the standard one does not work
    '''
    add_boolean_argument(p, 'usebestk', default=False)
    add_boolean_argument(p, 'segments',  default=False)
    add_boolean_argument(p, 'weighted_by_duration', default=False)
    add_boolean_argument(p, 'recon_flag', default=True)
    add_boolean_argument(p, 'verbose', default=False)
    add_boolean_argument(p, 'verbose_fact', default=False)
    add_boolean_argument(p, 'verbose_om', default=False)
    add_boolean_argument(p, 'normalize', default=True)
    add_boolean_argument(p, 'mask_flag', default=True)
    add_boolean_argument(p, 'test', default=False)
    
    return p
    
