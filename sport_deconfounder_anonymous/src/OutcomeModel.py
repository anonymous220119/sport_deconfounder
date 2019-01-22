
import numpy as np
import pandas as pd

import FactorLineups as fl

from sklearn import cross_validation
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score

class OutcomeModel :
	def __init__(self,df,FL,cols_home_score='home_team_goal',cols_away_score='away_team_goal',seed_cv=10,vad_portion=0.1,recon_flag=True,normalize=True,verbose=False,n_iter=10,n_iter_cv=10,min_caps=10,epsilon=0.1,weights=None):  

		self.verbose=verbose
		self.seed_cv=seed_cv
		self.vad_portion=vad_portion
		self.recon_flag=recon_flag
		self.normalize=normalize
		self.n_iter=n_iter
		self.n_iter_cv=n_iter_cv
		self.min_caps=min_caps
		self.epsilon=epsilon
		self.weights=weights
		self.cols_home_score=cols_home_score
		self.cols_away_score=cols_away_score

		self._initialize(df,FL)    # set outcome and causes

	def _set_players_caps(self,FL):
		self.idx2caps=dict((i,self.caps[i]) for i in range(FL.n_players) )
		self.api_id2caps=dict((FL.players[i],self.caps[i]) for i in range(FL.n_players) )
		try:
			self.player_name2caps=dict((FL.idx2player_name[i],self.caps[i]) for i in range(FL.n_players) ) # only there in case player name is different than api_id
		except:1

	def _filter_out_low_cap_players(self,FL):

		X0=np.copy(self.causes)
	
		X0[:,FL.api_id2idx[-1]]+=X0[:,np.where(self.caps<self.min_caps)[0]].sum(axis=1) # add removed players to the -1 dummy one
		X0[:,np.where(self.caps<self.min_caps)[0]]=0  # set to zero all games played by filtered-out players
		assert X0.sum(axis=1).all()==0

		return X0

	def _initialize(self,df,FL):  

		self.Y=(df[self.cols_home_score]-df[self.cols_away_score]).values
		self.Y_categorical=sharp_predict(self.Y,epsilon=self.epsilon)
		self.causes=FL._build_lineup_array(df,FL.cols_home) - FL._build_lineup_array(df,FL.cols_away) 
		if self.causes.shape[-1]!=FL.n_players: self.causes=np.append(self.causes,np.zeros((self.causes.shape[0],1)).astype(int),axis=1) # accounts for NaN or low cap players
		
		self.caps=np.sum(abs(self.causes),axis=0)
		self._set_players_caps(FL)
		if self.min_caps>0:
			self.causes=self._filter_out_low_cap_players(FL)


	# ------------------------------------------------------------------------------------------------------------------------
	# ----------  Functions needed to perform the cross validation routine ----------  
	# ------------------------------------------------------------------------------------------------------------------------

	def _infer_alpha_ridge_CV(self,solver='auto',alphas=None,vad_portion=0.1):

		vds = cross_validation.ShuffleSplit(self.X.shape[0], n_iter=self.n_iter_cv,test_size=vad_portion, random_state=self.seed_cv+1234) 

		if alphas==None:
			alphas=np.linspace(1, 19, 19, endpoint=True) # if you start from 0 you'll get ill-conditioning errors on output, due to sparse data
			alphas=np.append(alphas,np.linspace(10, 100, 10, endpoint=True))
			alphas=np.append(alphas,np.linspace(110, 200, 10, endpoint=True))

		model = Ridge(normalize=self.normalize,random_state=self.seed_cv,solver=solver)
		grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas),cv=vds)
		grid.fit(self.X,self.Y)
		if self.verbose:
			print(grid)
			print(grid.best_estimator_.alpha)
		return grid.best_estimator_.alpha

	# def _infer_alpha_ridge_CV(self,solver='auto',alphas=None):

	   #  rs = cross_validation.ShuffleSplit(self.X.shape[0], n_iter=self.n_iter_cv,test_size=self.holdout_portion, random_state=self.seed_cv)
	   #  for tr_idx, ts_idx in rs:
	   #      test_index=ts_idx
	   #      if vad_portion>0.:
	   #          vds = cross_validation.ShuffleSplit(len(tr_idx), n_iter=self.n_iter_cv,test_size=self.vad_portion, random_state=self.seed_cv+1) 
	   #          for vad_tr,vad_ts in vds:
	   #              train_index=tr_idx[vad_tr]
	   #              vad_index=tr_idx[vad_ts]
	   #      else:train_index=ts_idx
		
	   #  if alphas==None:
	   #      alphas=np.linspace(0, 19, 20, endpoint=True)
	   #      alphas=np.append(alphas,np.linspace(10, 100, 10, endpoint=True))
	   #      alphas=np.append(alphas,np.linspace(110, 200, 10, endpoint=True))

	   #  model = Ridge(normalize=self.normalize,random_state=self.seed_cv,solver=solver)
	   #  grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas),cv=vds)
	   #  grid.fit(self.X[tr_idx],self.Y[tr_idx])
	   #  if self.verbose:
	   #      print(grid)
	   #      # summarize the results of the grid search
	   #      # print(grid.best_score_)
	   #      print(grid.best_estimator_.alpha)
	   #  return grid.best_estimator_.alpha

	def _ridge_cv(self,solver='auto',alpha_ridge=None,alphas=None,holdout_portion=0.2):
		
		mae=[]; mse=[];acc=[]

		if alpha_ridge is None: alpha_ridge=self._infer_alpha_ridge_CV(solver=solver,alphas=alphas,vad_portion=holdout_portion)

		rs = cross_validation.ShuffleSplit(self.Y.shape[0], n_iter=self.n_iter,test_size=holdout_portion, random_state=self.seed_cv)

		for train_index, test_index in rs:
			
			regr = Ridge(alpha=alpha_ridge,normalize=self.normalize,random_state=self.seed_cv,solver=solver)
			if self.weights is not None:
				regr.fit(self.X[train_index], self.Y[train_index],sample_weight=self.weights[train_index])
			else:regr.fit(self.X[train_index], self.Y[train_index])
			y_pred_test = regr.predict(self.X[test_index])
			y_pred_train = regr.predict(self.X[train_index])
			y_pred=regr.predict(self.X)
			# intercept=regr.intercept_
			intercept=np.mean(self.Y[train_index],axis=0)
			
			if self.Y.ndim==2:
				for i in range(self.Y.shape[1]):
					mae.append(mae_output(self.Y[train_index][:,i],self.Y[test_index][:,i],y_pred_train[:,i],y_pred_test[:,i],intercept[i]))
					mse.append(mse_output(self.Y[train_index][:,i],self.Y[test_index][:,i],y_pred_train[:,i],y_pred_test[:,i],intercept[i]))
			else:
				mae.append(mae_output(self.Y[train_index],self.Y[test_index],y_pred_train,y_pred_test,intercept))
				mse.append(mse_output(self.Y[train_index],self.Y[test_index],y_pred_train,y_pred_test,intercept))

			if self.Y_categorical is not None:
				Y_pred_sharp=sharp_predict(y_pred,epsilon=epsilon,outcome=[-3,0,3])
				acc.append(accuracy_output(self.Y_categorical[train_index],self.Y_categorical[test_index],Y_pred_sharp[train_index],Y_pred_sharp[test_index],verbose=False))

		return np.array(mae),np.array(mse),np.array(acc)


	def estimate_alpha_ridge(self,confounder,solver='auto',alphas=None,test=False,holdout_portion=0.2):
		'''
		self should be an object representing a training set
		'''

		alpha_ridge={};

		for confunder_flag in [True, False]: 
		
			if confunder_flag==True:
				pred_name='causal'
				self.X =np.column_stack([self.causes, confounder])  # we use array instead of csr matrix because the sklearn implementation with normalized==True messes up with sparse matrices
			else:
				pred_name='non_causal'
				self.X=np.copy(self.causes)
			if self.verbose:print(confunder_flag)

			'''
			Estimate alpha, Ridge regularization parameter
			'''
			alpha_ridge[pred_name]=self._infer_alpha_ridge_CV(solver=solver,alphas=alphas,vad_portion=holdout_portion) # uses self.X and self.Y
			
			'''
			Optional: Test the model using Cross Validation
			'''
			if test==True:
				mae,mse,acc=self._ridge_cv(solver=solve,alpha_ridge=alpha_ridge[pred_name],holdout_portion=holdout_portion)
				
				if self.verbose:
					print('MAE test (fit vs baseline):',mae.mean(axis=0)[1],mae.mean(axis=0)[3])
					print('MSE test (fit vs baseline):',mse.mean(axis=0)[1],mse.mean(axis=0)[3])
					try:
						print('Accuracy test (fit vs baseline):',acc.mean(axis=0)[1],acc.mean(axis=0)[3])
					except: print('This outcome does not allow testing for total score accuracy')

		return alpha_ridge


def run_ridge(X,Y,alpha_ridge,normalize=True,seed=10,solver='auto'):
	regr = Ridge(alpha=alpha_ridge,normalize=normalize,random_state=seed,solver=solver)
	regr.fit(X, Y)
	return regr


def causal_noncausal_regr(X,Y,confounder,alpha_ridge,normalize=True,seed=10,solver='auto'):

	X_aug={};regr={}
	for model in alpha_ridge.keys():
		if model=='non_causal':
			X_aug[model] =np.column_stack([X, np.zeros(confounder.shape)]) 
		else:
			X_aug[model] =np.column_stack([X, confounder])

		regr[model]=run_ridge(X_aug[model],Y,alpha_ridge[model],normalize=normalize,seed=seed,solver=solver)

	return X_aug,regr


def mae_output(y_train,y_test,y_pred_train,y_pred_test,intercept,verbose=False):
	mae_tr=mean_absolute_error(y_train, y_pred_train) 
	mae_te=mean_absolute_error(y_test, y_pred_test)
	mae_tr_bs=mean_absolute_error(y_train, intercept*np.ones(len(y_train)))
	mae_te_bs=mean_absolute_error(y_test, intercept*np.ones(len(y_test)))
	if verbose:
		print('MAE train/test %.3f %.3f' %(mae_tr,mae_te))
		print('MAE baseline train/test %.3f %.3f' %(mae_tr_bs,mae_te_bs))

	return mae_tr,mae_te,mae_tr_bs,mae_te_bs

def mse_output(y_train,y_test,y_pred_train,y_pred_test,intercept,verbose=False):
	mse_tr=mean_squared_error(y_train, y_pred_train) 
	mse_te=mean_squared_error(y_test, y_pred_test)
	mse_tr_bs=mean_squared_error(y_train, intercept*np.ones(len(y_train)))
	mse_te_bs=mean_squared_error(y_test, intercept*np.ones(len(y_test)))
	if verbose:
		print('MSE train/test %.3f %.3f' %(mse_tr,mse_te))
		print('MSE baseline train/test %.3f %.3f' %(mse_tr_bs,mse_te_bs))

	return mse_tr,mse_te,mse_tr_bs,mse_te_bs


def sharp_predict(y_pred,epsilon=0.5,outcome=[-3,0,3]):
	y_pred_sharp=outcome[1]*np.ones(len(y_pred))
	if y_pred.ndim==2:
		y_pred_sharp[y_pred[:,0]-y_pred[:,1]<-epsilon]=outcome[0]
		y_pred_sharp[y_pred[:,0]-y_pred[:,1]>epsilon]=outcome[2]
	elif y_pred.ndim==1:
		y_pred_sharp[y_pred<-epsilon]=outcome[0]
		y_pred_sharp[y_pred>epsilon]=outcome[2]

	return y_pred_sharp

def accuracy_output(y_train,y_test,y_pred_train,y_pred_test,verbose=False):
	intercept=max(list(y_train),key=list(y_train).count)
	acc_tr=accuracy_score(y_train, y_pred_train) 
	acc_te=accuracy_score(y_test, y_pred_test)
	acc_tr_bs=accuracy_score(y_train, intercept*np.ones(len(y_train)))
	acc_te_bs=accuracy_score(y_test, intercept*np.ones(len(y_test)))
	if verbose:
		print('MSE train/test %.3f %.3f' %(acc_tr,acc_te))
		print('MSE baseline train/test %.3f %.3f' %(acc_tr_bs,acc_te_bs))

	return acc_tr,acc_te,acc_tr_bs,acc_te_bs

