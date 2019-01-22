"""
Factor model for the lineups
"""
import time
import sys,glob
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.preprocessing import MultiLabelBinarizer
import cv_fold as CV
import data_analysis_tools as da_tl

class FactorLineups :
    def __init__(self,df,cols_home=[],cols_away=[],cols_home_team='home_team_api_id',cols_away_team='away_team_api_id',seed_cv=10,K=1,segments=False,max_iter=400,tol=1e-08,S=1,alpha=0.1,verbose_fact=False,mask_flag=False,verbose=False):  

        self.verbose=verbose
        self.cols_home=cols_home
        self.cols_away=cols_away
        self.cols_home_team=cols_home_team
        self.cols_away_team=cols_away_team
        self.n_players=0
        self.n_teams=0
        self.players=[] # constains the classes of players, i.e. api_ids
        self.api_id2idx=dict()
        self.idx2api_id=dict()
        self.team_id2team_api_id=dict()
        self.team_api_id2team_id=dict()
        self.team_formation=[] # contains non-zero elements of the M-dim one-hot array for any given team
        self.team_per_player=[] # this is used only if you want to extract information for one specific player, optional. Not needed by the factorization algorithm
        self.mask=[]
        self.mask_flag=mask_flag # whether to multiply the reconstructed lineups for the mask or not
        self.seed_cv=seed_cv
        self.n_matches=len(df)
        self.segments=segments
        self.holdout_mask_test=[]
        self.holdout_mask_train=[]
        
        '''
        Variables needed for the inference of the factorization parameters 
        '''
        self.K=K
        self.max_iter=max_iter
        self.tol=tol
        self.S=S               # number of different random initializations for the factorization routine
        self.alpha=alpha       # alpha: shape parameter for Gamma prior in the bayesian case
        self.verbose_fact=verbose_fact # print verbose inside the facotirzation inference steps
        
        self._initialize(df)    # set player and team ids dict, team formations

    def _set_player_ids(self,df):
        " Assign id to each player api_id "
        mlb_all_players = MultiLabelBinarizer()
        all_players = df[self.cols_home+self.cols_away].values
        mlb_all_players.fit_transform(all_players)
        if -1 not in mlb_all_players.classes_:
            mlb_all_players.classes_=np.append(mlb_all_players.classes_,-1)
            assert mlb_all_players.classes_[-1]==-1

        self.players=mlb_all_players.classes_
        self.n_players=len(self.players)
        self.api_id2idx=dict((api,idx) for api,idx in zip(self.players,np.arange(self.n_players)))
        self.idx2api_id=dict((api,idx) for api,idx in zip(np.arange(self.n_players),self.players))


    def _set_team_ids(self,df):
        " Assign id to each team api_id"               
        self.teams=list(np.unique(df[[self.cols_home_team,self.cols_away_team]].values))
        self.n_teams=len(self.teams)
        self.team_id2team_api_id=dict((i,api) for i,api in zip(np.arange(self.n_teams),self.teams))
        self.team_api_id2team_id=dict((i,api) for i,api in zip(self.teams,np.arange(self.n_teams)))

    def set_players_names(self,df,col1='player_api_id',col2='player_name'):
        '''
        Optional: if one want to add players names from a separate dataframe
        '''
        self.api_id2player_name=dict((api,idx) for api,idx in zip(df[col1],df[col2]))
        self.player_name2api_id=dict((api,idx) for api,idx in zip(df[col2],df[col1]))
        if -1 not in self.api_id2player_name.keys():self.api_id2player_name[-1]='None'
        if 'None' not in self.player_name2api_id.keys():self.player_name2api_id['None']=-1

        self.idx2player_name=dict((idx,self.api_id2player_name[self.idx2api_id[idx]]) for idx in self.idx2api_id )
        self.player_name2idx=dict((self.api_id2player_name[self.idx2api_id[idx]],idx) for idx in self.idx2api_id )

    def set_team_names(self,df,col1='team_api_id',col2='team_long_name'):
        '''
        Optional: if one want to add team names from a separate dataframe
        '''
        self.team_api_id2team_name=dict((api,idx) for api,idx in zip(df[col1],df[col2]))
        self.team_name2team_api_id=dict((api,idx) for api,idx in zip(df[col2],df[col1]))

        self.team_id2team_name=dict((idx,self.team_api_id2team_name[self.team_id2team_api_id[idx]]) for idx in self.team_id2team_api_id)
        self.team_name2team_id=dict((self.team_api_id2team_name[self.team_id2team_api_id[idx]],idx) for idx in self.team_id2team_api_id)

    def teams_to_players(self):
        try:
            self.player_name2team_name=dict((self.idx2player_name[i],(',').join([self.team_id2team_name[self.team_per_player[i][j]] for j in range(len(self.team_per_player[i]))])) for i in range(self.n_players) )
        except:print('Not enough infos!')
            
    def output_formation(self,team_score,team_id=None,K=-1):
        '''
        Sorted by the score
        '''
        if team_id is not None:print(self.team_api_id2team_name[self.team_id2team_api_id[team_id]])
        return [self.api_id2player_name[self.idx2api_id[t]] for t in np.argsort(-team_score)[:K]]

    def _build_adjacency_by_team(self,df,team_type,cols):
        '''
        Team type is the opponent,reference lineup is the against
        B[a,i] is the number of time player i played against team a
        team_api_id2team_id is needed to have the correct correspondence team idx vs team_name
        C: TxM dim 
        '''
        grouped=df.groupby(by=team_type)
        B={}
        for name, group in grouped:
            team_idx=self.team_api_id2team_id[name]
            mlb_tmp = MultiLabelBinarizer(classes=self.players)
            try:
                a=group.loc[:,cols]
            except:
                raise Exception('cols is not valid')

            unique_elements, counts_elements = np.unique(a.values.ravel(), return_counts=True)
            tmp=mlb_tmp.fit_transform(unique_elements.reshape(1,-1))
            assert tmp.shape[0]==1
            for i in range(len(unique_elements)): # traded players
                if counts_elements[i]>1:
                    tmp[0,self.api_id2idx[unique_elements[i]]]=counts_elements[i]
            
            B[team_idx]=tmp[0]
        C=[ B[i] for i in range(len(B))]
        return np.array(C)

    def _extract_team_formations(self,df):
        '''
        For each team selects players' api ids that played at least once. This is the team formation.
        C_ij: # of times that player j played for team i
        '''
        C_home=self._build_adjacency_by_team(df,team_type=self.cols_home_team,cols=self.cols_home)
        C_away=self._build_adjacency_by_team(df,team_type=self.cols_away_team,cols=self.cols_away)
        C=C_home+C_away
        team_formation=[]
        mask=np.zeros(C.shape).astype(int)
        for i in range(C_home.shape[0]):
            team_formation.append(np.nonzero(C[i])[0])
            mask[i][team_formation[-1]]=1
        team_per_player=[]
        for i in range(C.shape[1]):
            team_per_player.append(np.nonzero(C[:,i])[0])
        
        self.team_formation=team_formation
        self.mask=mask
        self.team_per_player=team_per_player

    def _initialize(self,df):         
        
        self._set_player_ids(df)
        self._set_team_ids(df)
        self._extract_team_formations(df)


    def _build_lineup_array(self,df,cols):
        '''
        From a dataframe with player names into a one-hot encoded matrix NxM, N=number of matches, M= number of players
        '''
        mlb = MultiLabelBinarizer(classes=self.players)
        on_court = df[cols].values
        on_court = pd.DataFrame(mlb.fit_transform(on_court), 
                                     columns=mlb.classes_, 
                                     index=df.index)
        return on_court.values

    def team_pairs(self):
        '''
        (a,b) and (b,a) count as one, i.e. order does not matter. It only matters if one distinguishes home vs away
        '''
        pairs=[]
        for i in range(self.n_teams):
            for j in range(i+1,self.n_teams):
                pairs.append((self.teams[i],self.teams[j]))
        return pairs

    def match_by_pairs(self,df,pairs):
        '''
        Group dataframe by unordered pairs of teams
        '''
        grouped_by_pairs={}
        for a,b in pairs:
            try:
                grouped_by_pairs[(a,b)]=df[(df[self.cols_home_team].isin([a,b])) & (df[self.cols_away_team].isin([a,b]))]
            except:
                print((a,b), ' not in df') 
        return grouped_by_pairs

    # ------------------------------------------------------------------------------------------------------------------------
    # ----------  Functions needed to perform the cross validation routine ----------  
    # ------------------------------------------------------------------------------------------------------------------------

    def _stratified_split(self,df,holdout_portion=0.2,seed=10,with_repetitions=True):
        '''
        Stratified: divided the games by pair of teams (a,b), and select these pairs btw train/test. 
        with_repetitions=False: In this way the same teams will not appear in both sets 
        Particularly relevant if working with game segments.
        '''
        if with_repetitions==True:

            grouped=df.groupby(by=[self.cols_home_team,self.cols_away_team])
            N=len(grouped.groups.keys())# pairs of teams
            
            prng=np.random.RandomState(seed)
            permuted=prng.permutation(N)
            test=permuted[:int(holdout_portion * N)]
            train=permuted[int(holdout_portion * N):]

            '''
            Optional: save the holdout mask, can be useful for future functions
            '''
            holdout_mask=np.zeros((self.n_teams,self.n_teams,self.n_players),dtype=bool)
            for i in range(len(test)):
                a_id,b_id=grouped.groups.keys()[test[i]]
                a=self.team_api_id2team_id[a_id]
                b=self.team_api_id2team_id[b_id]
                holdout_mask[a,b]=True
                holdout_mask[b,a]=True # symmetric
            self.holdout_mask_test=holdout_mask
            self.holdout_mask_train=~holdout_mask
            for i in range(self.n_teams):
                if holdout_mask[i].sum()==0:print('team not in the test set',i)
                if self.holdout_mask_train[i].sum()==0:print('team not in the training set',i)

            test_idx = np.zeros(N, dtype=bool)
            test_idx[test] = True

            test_df_index=list(np.array(grouped.groups.values())[test_idx])
            train_df_index=list(np.array(grouped.groups.values())[~test_idx])

            test_df = pd.concat([df.loc[test_df_index[i]] for i in range(len(test_df_index))])
            train_df = pd.concat([df.loc[train_df_index[i]] for i in range(len(train_df_index))])

            return test_df,train_df

        else:
            pairs=self.team_pairs()
            grouped_by_pairs=self.match_by_pairs(df,pairs)
            N=len(pairs)
            assert N>0

            prng=np.random.RandomState(seed)
            permuted=prng.permutation(N)
            test=permuted[:int(holdout_portion * N)]
            train=permuted[int(holdout_portion * N):]

            test_pairs=[pairs[test[i]] for i in range(len(test))]
            train_pairs=[pairs[train[i]] for i in range(len(train))]

            test_df=pd.concat([grouped_by_pairs[pair] for pair in test_pairs ])
            train_df=pd.concat([grouped_by_pairs[pair] for pair in train_pairs ])

            '''
            Optional: save the holdout mask, can be useful for future functions
            '''
            self.test_pairs=[]
            holdout_mask=np.zeros((self.n_teams,self.n_teams,self.n_players),dtype=bool)
            for i in range(len(test_pairs)):
                a_api_id,b_api_id=test_pairs[i]
                a=self.team_api_id2team_id[a_api_id]
                b=self.team_api_id2team_id[b_api_id]
                holdout_mask[a,b]=True
                holdout_mask[b,a]=True # symmetric
            self.holdout_mask_test=holdout_mask
            self.holdout_mask_train=~holdout_mask
            for i in range(self.n_teams):
                if holdout_mask[i].sum()==0:print 'team not in the test set',i
                if self.holdout_mask_train[i].sum()==0:print 'team not in the training set',i

            return test_df,train_df

    def _single_split(self,df,holdout_portion=0.2,seed=10,stratified=False,with_repetitions=True):
        
        if stratified:test_df,train_df=self._stratified_split(df,holdout_portion=holdout_portion,seed=seed,with_repetitions=with_repetitions)
        else:
            cvfm=CV.cv_fold_fm(holdout_portion=holdout_portion, random_state=seed)
            test_df,train_df=cvfm.split(df)

        return test_df,train_df

    def _tensorial_representation(self,df,segments=False,weighted_by_duration=False):
        
        N=len(df)   # number of matches
        T=self.n_teams
        M=self.n_players

        home_teams_per_match=df[self.cols_home_team].values
        away_teams_per_match=df[self.cols_away_team].values

        home_lineup=self._build_lineup_array(df,self.cols_home)
        away_lineup=self._build_lineup_array(df,self.cols_away)

        A=np.zeros((T,T,M)).astype(int)
        holdout_mask=np.zeros(A.shape).astype(int)
        for t in range(T):holdout_mask[t,np.arange(T)]=np.copy(self.mask[t])
        if segments:elapsed=df['elapsed_diff'].values
        for i in range(N):  # cycle over the matches
            t1=int(self.team_api_id2team_id[home_teams_per_match[i]])
            t2=int(self.team_api_id2team_id[away_teams_per_match[i]]) 
            if segments:  
                if weighted_by_duration==True: 
                    A[t1,t2]+=(elapsed[i]*home_lineup[i]).astype(int)
                    A[t2,t1]+=(elapsed[i]*away_lineup[i]).astype(int)
                else:
                    A[t1,t2]+=(home_lineup[i]).astype(int)
                    A[t2,t1]+=(away_lineup[i]).astype(int)
            else:
                A[t1,t2]+=np.copy(home_lineup[i].astype(int))
                A[t2,t1]+=np.copy(away_lineup[i].astype(int))

        B=A.sum(axis=0)   # depends on the opponent only   B[a,i]= # times that player i played against team a

        return A,B,holdout_mask

    def run_factorization(self,df,algos=[],weighted_by_duration=False):

        A,B,holdout_mask=self._tensorial_representation(df,segments=self.segments,weighted_by_duration=weighted_by_duration)

        Z=dict()
        A_recon=dict()
        fact_obj=dict()

        if 'BPTF' in algos:
            '''
            MAP : BPTF, Bayesian
            '''
            fact_obj['BPTF']=da_tl.run_bptf(A,holdout_mask=None,K=self.K,max_iter=self.max_iter,verbose=self.verbose_fact,tol=self.tol,alpha=self.alpha,seed0=self.seed_cv,S=self.S)

            # fact_obj['BPTF']=da_tl.run_bptf(A,holdout_mask=self.holdout_mask_train,K=self.K,max_iter=self.max_iter,verbose=self.verbose_fact,tol=self.tol,alpha=self.alpha,seed0=self.seed_cv,S=self.S)

            Z['BPTF']=np.einsum('pk,tk->ptk',fact_obj['BPTF'].G_DK_M[0],fact_obj['BPTF'].G_DK_M[1])  # substitute deconfounder
            A_recon['BPTF']=np.einsum('ptk,ik->pti',Z['BPTF'],fact_obj['BPTF'].G_DK_M[2])  # reconstructed causes
        
        if 'MT' in algos:
            '''
            MLE : MultiTensor, frequentist
            '''
            fact_obj['MT']=da_tl.run_MultiTensor(A,K=self.K,max_iter=self.max_iter,verbose=self.verbose_fact,tol=self.tol,seed=self.seed_cv,assortative=True,N_real=self.S)

            Z['MT']=np.einsum('ak,bk->abk',fact_obj['MT'].u_f,fact_obj['MT'].v_f)  # substitute deconfounder
            A_recon['MT']=np.einsum('abk,ki->abi',Z['MT'],fact_obj['MT'].w_f)  # reconstructed causes  

        if 'BPMF' in algos:
            '''
            MAP : BPTF, Bayesian
            '''
            fact_obj['BPMF']=da_tl.run_bptf(B,K=self.K,max_iter=self.max_iter,verbose=self.verbose_fact,tol=self.tol,alpha=self.alpha,seed0=self.seed_cv,S=self.S)

            Z['BPMF']=fact_obj['BPMF'].G_DK_M[0]  # substitute deconfounder, first dim is the opponent team
            A_recon['BPMF']=np.einsum('ak,ik->ai',Z['BPMF'],fact_obj['BPMF'].G_DK_M[1])  # reconstructed causes       


        return Z,A_recon,fact_obj

    def _substitute_confounder(self,df,Z):
        '''
        Use the substitute confounder: dimension is  NxK 
        '''   
        N=len(df)   # number of matches in test set
        algos=Z.keys()
        scf=dict()

        home_teams_per_match=df[self.cols_home_team].values
        away_teams_per_match=df[self.cols_away_team].values

        home_lineup=self._build_lineup_array(df,self.cols_home)
        away_lineup=self._build_lineup_array(df,self.cols_away)

        for algo in algos:
            assert Z[algo].shape[-1]==self.K
            scf[algo]=dict()
            for a in ['home','away']: # denotes the reference lineup
                scf[algo][a]=np.zeros((N,self.K))
                for i in range(N):  # match index
                    idx1=self.team_api_id2team_id[home_teams_per_match[i]]
                    idx2=self.team_api_id2team_id[away_teams_per_match[i]]
                    if a=='home':
                        if Z[algo].ndim==3:
                            scf[algo][a][i]=Z[algo][idx1,idx2]
                        elif Z[algo].ndim==2:
                            scf[algo][a][i]=Z[algo][idx2]
                    else:
                        if Z[algo].ndim==3:
                            scf[algo][a][i]=Z[algo][idx2,idx1]
                        elif Z[algo].ndim==2:
                            scf[algo][a][i]=Z[algo][idx1]
                        
        return scf

    def _reconstruct_lineups(self,df,A_recon):
        '''
        Use the reconstructed lineup: dimension is  NxM
        ''' 
        
        N=len(df)   # number of matches in test set
        algos=A_recon.keys()
        reconstructed=dict()

        home_teams_per_match=df[self.cols_home_team].values
        away_teams_per_match=df[self.cols_away_team].values

        home_lineup=self._build_lineup_array(df,self.cols_home)
        away_lineup=self._build_lineup_array(df,self.cols_away)

        for algo in algos:
            reconstructed[algo]=dict()
            for a in ['home','away']: # denotes the reference lineup
                reconstructed[algo][a]=np.zeros((N,self.n_players))
                for i in range(N):  # match index
                    idx1=self.team_api_id2team_id[home_teams_per_match[i]]
                    idx2=self.team_api_id2team_id[away_teams_per_match[i]]
                    if a=='home':
                        
                        if A_recon[algo].ndim==3:
                            if self.mask_flag==True:
                                reconstructed[algo][a][i]=A_recon[algo][idx1,idx2]*self.mask[idx1] 
                            else:
                                reconstructed[algo][a][i]=A_recon[algo][idx1,idx2]
                        elif A_recon[algo].ndim==2:
                            if self.mask_flag==True:
                                reconstructed[algo][a][i]=A_recon[algo][idx2]*self.mask[idx1] 
                            else:
                                reconstructed[algo][a][i]=A_recon[algo][idx2]
                    else:
                        if A_recon[algo].ndim==3:
                            if self.mask_flag==True:
                                reconstructed[algo][a][i]=A_recon[algo][idx2,idx1]*self.mask[idx2] 
                            else:
                                reconstructed[algo][a][i]=A_recon[algo][idx2,idx1]
                        elif A_recon[algo].ndim==2:
                            if self.mask_flag==True:
                                reconstructed[algo][a][i]=A_recon[algo][idx1]*self.mask[idx2] 
                            else:
                                reconstructed[algo][a][i]=A_recon[algo][idx1]
        return reconstructed

    def _observed_lineups(self,df):
        '''
        Observed lineups home and away
        '''
        N=len(df)

        observed_teams=dict()
        observed_lineup=dict()

        observed_teams['home']=df[self.cols_home_team].values
        observed_teams['away']=df[self.cols_away_team].values

        observed_lineup['home']=self._build_lineup_array(df,self.cols_home)
        observed_lineup['away']=self._build_lineup_array(df,self.cols_away)

        return observed_teams,observed_lineup

    def transform(self,A,df):
        '''
        A: TxM array: A[t,pl]=score of player pl if playing in team t. E.g. to calculate most frequent team formation
        OUTPUT: transformed: dict with keys 'home' and 'away', each N x M dimensional, rows are games
        '''
        N=len(df)

        transformed_lineup=dict()
        observed_teams=dict()

        observed_teams['home']=df[self.cols_home_team].map(self.team_api_id2team_id).values
        observed_teams['away']=df[self.cols_away_team].map(self.team_api_id2team_id).values

        for a in observed_teams.keys():
            transformed_lineup[a]=np.zeros((N,self.n_players))
            for i in range(N):
                transformed_lineup[a][i]=A[observed_teams[a][i]]

        return transformed_lineup
                
    def fit(self,df,Z,A_recon,recon_flag=True):

        if recon_flag==True: # use reconstructed causes
            transformed=self._reconstruct_lineups(df,A_recon)
        else:
            print('Using the sub confounder')
            transformed=self._substitute_confounder(df,Z)
        return transformed

    
    def performance_evaluation(self,observed_teams,pred_lineup,observed_lineup,at_K=11,at_K2=20,output_mean=False,K=-1,fold=-1):
        
        prng=np.random.RandomState(self.seed_cv) # needed for the random extraction of 11 players from team formation
        '''
        Calculates by default AUC, precision@K and ncdg@k, game-by-game
        '''
        N=len(observed_lineup['home'])
        algos=pred_lineup.keys()

        '''
        Set up variables to store results
        '''
        auc={};
        precision_at11={};ncdg_at11={}
        precision_at20={};ncdg_at20={}

        precision_at11['rand']={};precision_at20['rand']={}
        
        for algo in algos:
            auc[algo]={};
            precision_at11[algo]={};ncdg_at11[algo]={}
            precision_at20[algo]={};ncdg_at20[algo]={}
        
            for a in ['home','away']:
                auc[algo][a]=[];
                precision_at11[algo][a]=[];ncdg_at11[algo][a]=[]
                precision_at20[algo][a]=[];ncdg_at20[algo][a]=[]

        for a in ['home','away']:

            precision_at11['rand'][a]=[];precision_at20['rand'][a]=[]

            for algo in algos:
                auc[algo][a]=[];
                precision_at11[algo][a]=[];ncdg_at11[algo][a]=[]
                precision_at20[algo][a]=[];ncdg_at20[algo][a]=[]

            for i in range(N):
                y_true=observed_lineup[a][i].flatten()
                nnz_true=np.arange(len(y_true))
                for algo in algos:
                    
                    if self.mask_flag==True:
                        nnz=np.nonzero(pred_lineup[algo][a][i].flatten())[0]
                    else:
                        nnz=nnz_true
                    if len(nnz)>0:
                        result_i=performance_evaluation_single_game(pred_lineup[algo][a][i].flatten()[nnz],y_true[nnz],at_K=at_K,at_K2=at_K2)
                    
                        auc[algo][a].append(result_i[0])
                        precision_at11[algo][a].append(result_i[1]);precision_at20[algo][a].append(result_i[2])
                        ncdg_at11[algo][a].append(result_i[3]);ncdg_at20[algo][a].append(result_i[4])

                '''
                Random pick of 11 players over the team formation
                '''
                pl_lineup=set(np.nonzero(observed_lineup[a][i])[0])
                tid=self.team_api_id2team_id[observed_teams[a][i]]
                c=[];d=[]
                for s in range(self.S):
                    prng.shuffle(self.team_formation[tid]) # random pick of 11 players from the available ones in that team
                    c.append(len(pl_lineup.intersection(set(self.team_formation[tid][:at_K]))))
                    d.append(len(pl_lineup.intersection(set(self.team_formation[tid][:at_K2]))))

                precision_at11['rand'][a].append(np.array(c).mean())
                precision_at20['rand'][a].append(np.array(d).mean())
        results= (auc,precision_at11,precision_at20,ncdg_at11,ncdg_at20)
        if output_mean==True:
            return self._extract_means_from_results(results,K=K,fold=fold)
        else:
            return 1,results

    def _extract_means_from_results(self,results,metric_name=['auc','precision_at11','precision_at20','ncdg_at11','ncdg_at20'],K=-1,fold=-1):
        '''
        Results: zip(auc,precision_at11,precision_at20,ncdg_at11,ncdg_at20)
        Each one is a dict indexed by the algorithm's name, e.g. BPTF, MT, etc...
        Each dict is a list with one entry per game
        OUTPUT: mean across games
        '''
        results_mean=dict()
        for a in ['home','away']:
            results_mean[a]=[K,fold,a]
            results_name=['K','fold','lineup_type']
            for i in range(len(results)):
                metric=results[i]
                res_per_metric=dict()
                algos=metric.keys()
                for algo in algos:
                    try:
                        res_per_metric[algo]=np.array(metric[algo][a]).mean()
                    except: res_per_metric[algo]=None
                    results_mean[a].append(res_per_metric[algo])
                    results_name.append(metric_name[i]+'_'+algo)

        return results_name,results_mean


    def cross_validation(self,df,holdout_portion=0.2,F=10,algos=[],recon_flag=True,at_K=11,at_K2=20,outfile='cross_validation',stratified=False,with_repetitions=True,weighted_by_duration=False):
        '''
        Cross validation: split dataset into train/test
        '''
        self._initialize(df)    # set player and team ids dict, team formations

        # results={'home':[],'away':[]}
        results=[]
        for f in range(F):

            #for a in ['home','away']:results.append([self.K,f])

            test_df,train_df=self._single_split(df,holdout_portion=holdout_portion,seed=self.seed_cv+f,stratified=stratified,with_repetitions=with_repetitions)
            A,B,holdout_mask=self._tensorial_representation(train_df,segments=self.segments) # TODO: only needed to build A and use it as input to most_frequent_formation: need to make thismore efficent
            # best_team_formation=most_frequent_formation(A)    #  baseline

            Z,A_recon,fact_obj=self.run_factorization(train_df,algos=algos,weighted_by_duration=weighted_by_duration)
            pred_test=self.fit(test_df,Z,A_recon,recon_flag=recon_flag)
            pred_test['best_team_formation']=self.transform(A.sum(axis=1),test_df)  #  baseline

            observed_teams,observed_lineup=self._observed_lineups(test_df)

            results_name,results_f=self.performance_evaluation(observed_teams,pred_test,observed_lineup,at_K=at_K,at_K2=at_K2,output_mean=True,K=self.K,fold=f)
            
            for a in ['home','away']:results.append(results_f[a])

            if self.verbose:
                for a in ['home','away']:print(results_f[a])

        df_res=output_results_onfile(results,outfile,cols_res=results_name)

        return df_res




'''
------------------------------------------------------------------------------------------------------------
    Valid outside the class: functions that can be called for other purposes than e.g. cross-validation
------------------------------------------------------------------------------------------------------------
'''

def output_results_onfile(results,outfile,cols_res=None):
    
    if cols_res is None:
        algos=results[0].keys()
        cols_res=['K','fold']
        for algo in algos:cols_res.append('auc_'+algo)
        for algo in algos:cols_res.append('precision_at11_'+algo)
        for algo in algos:cols_res.append('precision_at20_'+algo)
        cols_res.append('precision_at11_rand');cols_res.append('precision_at20_rand')
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

def performance_evaluation_single_game(y_pred,y_true,at_K=11,at_K2=20):
        roc=zip(y_pred,y_true)
        roc.sort(key=lambda x: x[0], reverse=False);
        try:
            c,b=zip(*roc)
            Pos=np.count_nonzero(np.array(b))
            Neg=len(c)-Pos
            auc=da_tl.calculate_AUC(roc,Pos,Neg)
            precision_at11=sum([s for j,s in roc[-at_K:]])   # the top k players predicted in the lineup
            precision_at20=sum([s for j,s in roc[-at_K2:]])  # the top k players predicted in the lineup
            
            roc.sort(key=lambda x: x[1], reverse=True);
            r,b=zip(*roc)
            ncdg_at11=da_tl.ndcg_at_ki(r, k=at_K, method=0)
            ncdg_at20=da_tl.ndcg_at_ki(r, k=at_K2, method=0)
            return (auc,precision_at11,precision_at20,ncdg_at11,ncdg_at20)
        except:
            roc=zip(y_pred,y_true)
            roc.sort(key=lambda x: x[0], reverse=False);
            precision_at11=sum([s for j,s in roc[-at_K:]])   # the top k players predicted in the lineup
            precision_at20=sum([s for j,s in roc[-at_K2:]])  # the top k players predicted in the lineup
            
            roc.sort(key=lambda x: x[1], reverse=True);
            r,b=zip(*roc)
            ncdg_at11=da_tl.ndcg_at_ki(r, k=at_K, method=0)
            ncdg_at20=da_tl.ndcg_at_ki(r, k=at_K2, method=0)
            # precision_at11=precision_at20=ncdg_at11=ncdg_at20=None
            auc=0.5
            return (auc,precision_at11,precision_at20,ncdg_at11,ncdg_at20)
            # return (None,None,None,None,None)

                  
def most_frequent_formation(A):
    best_team_formation=(A.sum(axis=1))  # sum over the opponents
    normalization=best_team_formation.sum(axis=1).astype(float) # sum over the players
    best_team_formation=best_team_formation/normalization[:,np.newaxis] 
    return best_team_formation

