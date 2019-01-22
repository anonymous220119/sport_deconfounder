
import bptf as bptf
import MultiTensor as mt
from sklearn import cross_validation# import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error,accuracy_score
import numpy as np
import pandas as pd

def run_bptf(x_train,holdout_mask=None,K=10,max_iter=100,verbose=False,tol=1e-8,alpha=0.3,seed0=10,S=1):
    
    best_seed=-10
    elbo_max=-1000000000000
    prng=np.random.RandomState(seed0)
    s0=prng.randint(1234)
    for seed in range(s0,S+s0):
        ptf = bptf.BPTF(n_modes=x_train.ndim, n_components=K,max_iter=max_iter,verbose=verbose,tol=tol,alpha=alpha,seed=seed0)
        try:ptf.fit(x_train.toarray(), mask=holdout_mask)
        except:ptf.fit(x_train, mask=holdout_mask)
        if ptf.elbo[-1]>elbo_max:
            elbo_max=ptf.elbo[-1]
            best_seed=seed
        if verbose:print seed,ptf.elbo[-1],best_seed
    if verbose:print('best seed=', best_seed,' ELBO:',elbo_max)
    ptf=bptf.BPTF(n_modes=x_train.ndim, n_components=K,max_iter=max_iter,verbose=verbose,tol=tol,alpha=alpha,seed=best_seed)
    try:ptf.fit(x_train.toarray(), mask=holdout_mask)
    except:ptf.fit(x_train, mask=holdout_mask)

    return ptf

def run_MultiTensor(A,K=10,max_iter=300,verbose=False,tol=1e-8,seed=10,assortative=True,N_real=1,err_max=1e-7):
    ''''
    A_pth: p=player; t=team1;h=team2 --> 'layers' are players, 'nodes' are teams
    B[l,:,:]=nx.to_numpy_matrix(A[l],weight='weight'). --> this is our courrent tensor A[i,a,b]
    '''
    if A.ndim==3:
        B=np.einsum('abi->iab',A)
    else:
        B=A[np.newaxis,:,:]
        B=np.einsum('abi->iab',B)
    n_teams=B.shape[2]
    n_teams2=B.shape[1]
    n_players=B.shape[0]
    u_list=np.arange(n_teams2)  
    v_list=np.arange(n_teams)  
    MT=mt.MultiTensor(  N=n_teams, N2=n_teams2,L=n_players,K=K, N_real=N_real,tolerance=tol,maxit=max_iter,rseed=seed,assortative=bool(assortative),verbose=verbose,err_max=err_max)
    MT.cycle_over_realizations(B,u_list,v_list)   

    return MT

    
def infer_alpha_ridge_CV(X,Y,holdout_portion=0.2,vad_portion=0.1,seed=1,normalize=False,n_iter=1,solver='auto',verbose=False,alphas=None):
    rs = cross_validation.ShuffleSplit(X.shape[0], n_iter=1,test_size=holdout_portion, random_state=seed)
    for tr_idx, ts_idx in rs:
        test_index=ts_idx
        if vad_portion>0.:
            vds = cross_validation.ShuffleSplit(len(tr_idx), n_iter=n_iter,test_size=vad_portion, random_state=seed+1) 
            for vad_tr,vad_ts in vds:
                train_index=tr_idx[vad_tr]
                vad_index=tr_idx[vad_ts]
        else:train_index=ts_idx
    
    if alphas==None:
        alphas=np.linspace(0, 19, 20, endpoint=True)
        alphas=np.append(alphas,np.linspace(10, 100, 10, endpoint=True))
        alphas=np.append(alphas,np.linspace(110, 200, 10, endpoint=True))

    model = Ridge(normalize=normalize,random_state=seed,solver=solver)
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas),cv=vds)
    grid.fit(X[tr_idx],Y[tr_idx])
    if verbose:
        print(grid)
        # summarize the results of the grid search
        # print(grid.best_score_)
        print(grid.best_estimator_.alpha)
    return grid.best_estimator_.alpha

def ridge_cv(X,Y,Y_class=None,n_iter=1,seed=10,holdout_portion=0.2,normalize=False,solver='auto',alpha_ridge=None,vad_portion=0.1,epsilon=0.5,weights=None):
    mae=[]
    mse=[]
    acc=[]
    rs = cross_validation.ShuffleSplit(Y.shape[0], n_iter=n_iter,test_size=holdout_portion, random_state=seed)
    for train_index, test_index in rs:
        
        if alpha_ridge==None:
            vds = cross_validation.ShuffleSplit(len(train_index), n_iter=n_iter,test_size=vad_portion, random_state=seed+1) 
            alphas=np.linspace(0, 19, 20, endpoint=True)
            alphas=np.append(alphas,np.linspace(10, 100, 10, endpoint=True))
            alphas=np.append(alphas,np.linspace(110, 200, 10, endpoint=True))
            model = Ridge(normalize=normalize,random_state=seed,solver=solver)
            grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas),cv=vds)
            grid.fit(X[train_index],Y[train_index])
            alpha_ridge=grid.best_estimator_.alpha

        regr = Ridge(alpha=alpha_ridge,normalize=normalize,random_state=seed,solver=solver)
        if weights is not None:
            regr.fit(X[train_index], Y[train_index],sample_weight=weights[train_index])
        else:regr.fit(X[train_index], Y[train_index])
        y_pred_test = regr.predict(X[test_index])
        y_pred_train = regr.predict(X[train_index])
        y_pred=regr.predict(X)
        # intercept=regr.intercept_
        intercept=np.mean(Y[train_index],axis=0)
        
        if Y.ndim==2:
            for i in range(Y.shape[1]):
                mae.append(mae_output(Y[train_index][:,i],Y[test_index][:,i],y_pred_train[:,i],y_pred_test[:,i],intercept[i]))
                mse.append(mse_output(Y[train_index][:,i],Y[test_index][:,i],y_pred_train[:,i],y_pred_test[:,i],intercept[i]))
        else:
            mae.append(mae_output(Y[train_index],Y[test_index],y_pred_train,y_pred_test,intercept))
            mse.append(mse_output(Y[train_index],Y[test_index],y_pred_train,y_pred_test,intercept))

        if Y_class is not None:
            Y_pred_sharp=sharp_predict(y_pred,epsilon=epsilon,outcome=[-3,0,3])
            acc.append(accuracy_output(Y_class[train_index],Y_class[test_index],Y_pred_sharp[train_index],Y_pred_sharp[test_index],verbose=False))

    return np.array(mae),np.array(mse),np.array(acc)

def extract_coeff_linear_regression(lm,labels=None,rounding=4):
    myDF3 = pd.DataFrame()
    
    params=[]
    for i in range(len(lm)):
        if lm[i].coef_.ndim>1:  
            for j in range(lm[i].coef_.shape[0]):
                params.append(np.append(lm[i].intercept_[j],lm[i].coef_[j]))
                params[-1] = np.round(params[-1],rounding)
                myDF3["Coef"+str(i)+'_'+str(j)] = params[-1]
        else:
            params.append(np.append(lm[i].intercept_,lm[i].coef_))
            params[-1] = np.round(params[-1],rounding)
            myDF3["Coef"+str(i)] = params[-1]

    if labels is not None:
        myDF3["player_name"] = labels

    return myDF3

def sharp_predict(y_pred,epsilon=0.5,outcome=[-3,0,3]):
    y_pred_sharp=outcome[1]*np.ones(len(y_pred))
    if y_pred.ndim==2:
        y_pred_sharp[y_pred[:,0]-y_pred[:,1]<-epsilon]=outcome[0]
        y_pred_sharp[y_pred[:,0]-y_pred[:,1]>epsilon]=outcome[2]
    elif y_pred.ndim==1:
        y_pred_sharp[y_pred<-epsilon]=outcome[0]
        y_pred_sharp[y_pred>epsilon]=outcome[2]

    return y_pred_sharp

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

def accuracy_output(y_train,y_test,y_pred_train,y_pred_test,verbose=False):
    intercept=max(list(y_train),key=list(y_train).count)
    acc_tr=accuracy_score(y_train, y_pred_train) 
    acc_te=accuracy_score(y_test, y_pred_test)
    acc_tr_bs=accuracy_score(y_train, intercept*np.ones(len(y_train)))
    acc_te_bs=accuracy_score(y_test, intercept*np.ones(len(y_test)))
    if verbose:
        print('MSE train/test %.3f %.3f' %(mse_tr,mse_te))
        print('MSE baseline train/test %.3f %.3f' %(mse_tr_bs,mse_te_bs))

    return acc_tr,acc_te,acc_tr_bs,acc_te_bs

def calculate_AUC(M,Pos,Neg):
    # M= # SORTED (from small to big) List of 2-tuple, each entry is M[n]=(mu_ij,A_ij)      
    # Pos= # positive entries       
    # Neg= # negative entries 
    y=0.;bad=0.;
    for m,a in M:
        if(a>0):
            y+=1;
        else:
            bad+=y; 
    AUC=1.-(bad/(Pos*Neg));     
    return AUC;


def ndcg_at_ki(r, k=5, method=0):

    dcg_max = dcg_at_ki(sorted(r, reverse=True), k=k, method=min(1,method))
    # print(dcg_max)
    if not dcg_max:
        return 0.
    return dcg_at_ki(r, k=k, method=method) / dcg_max

def dcg_at_ki(r, k=5, method=0):
    r = np.asfarray(r)[:k]
    # r = r[:k]
    
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        elif method == 2:
            np.random.shuffle(r)
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.
