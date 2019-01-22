"""
Poisson Tensor factorization for Multi-layer networks.
Adapted from C De Bacco's code: https://github.com/cdebacco/MultiTensor
Uses an EM-like algorithm with uniform priors: MLE estimation
"""
import time
import sys
import numpy as np
from numpy.random import RandomState

class MultiTensor :
    def __init__(self,N=100,N2=None,L=1,K=2, N_real=1,tolerance=0.1,decision=10,maxit=500,rseed=0,inf=1e10,err_max=1e-100,err=0.1,undirected=False,assortative=False,verbose=False):  
        self.N=N
        if N2 is None: self.N2=N
        else:self.N2=N2
        self.L=L
        self.K=K
        self.N_real=N_real
        self.tolerance=tolerance
        self.decision=decision
        self.maxit=maxit
        self.rseed=rseed
        self.inf=inf
        self.err_max=err_max
        self.err=err
        self.undirected=undirected
        self.assortative=assortative
        self.maxL=-inf
        self.verbose=verbose

        # Values used inside the update
        self.u=np.zeros((self.N2,self.K),dtype=float)  # Out-going membership
        self.v=np.zeros((self.N,self.K),dtype=float)  # In-going membership

        # Old values
        self.u_old=np.zeros((self.N2,self.K),dtype=float)  # Out-going membership
        self.v_old=np.zeros((self.N,self.K),dtype=float)  # In-going membership
        # Final values after convergence --> the ones that maximize Likelihood
        self.u_f=np.zeros((self.N2,self.K),dtype=float)  # Out-going membership
        self.v_f=np.zeros((self.N,self.K),dtype=float)  # In-going membership

        if(self.assortative==True): # Purely diagonal matrix
            self.w=np.zeros((self.K,self.L),dtype=float)  # Affinity Matrix
            self.w_old=np.zeros((self.K,self.L),dtype=float)  # Affinity Matrix
            self.w_f=np.zeros((self.K,self.L),dtype=float)  # Affinity Matrix
        else:
            self.w=np.zeros((self.K,self.K,self.L),dtype=float)  # Affinity Matrix
            self.w_old=np.zeros((self.K,self.K,self.L),dtype=float)  # Affinity Matrix
            self.w_f=np.zeros((self.K,self.K,self.L),dtype=float)  # Affinity Matrix


    def _randomize_w(self,rng):
        " Assign a random number in (0,1.) to each entry"
        for i in range(self.L):
            for k in range(self.K):
                if(self.assortative==True):self.w[k,i]=rng.random_sample(1)
                else:
                    for q in range(k,self.K):
                        if(q==k):self.w[k,q,i]=rng.random_sample(1)
                        else: self.w[k,q,i]=self.w[q,k,i]=self.err*rng.random_sample(1)


    def _randomize_u_v(self,rng,u_list,v_list):
        " Randomize the memberships' entries different from zero"               
        rng=np.random.RandomState(self.rseed)   # Mersenne-Twister random number generator
        for k in range(self.K):
            for i in range(len(u_list)):
                j=u_list[i]
                self.u[j][k]=rng.random_sample(1)
                if(self.undirected==True):self.v[j][k]=self.u[j][k]
            if(self.undirected==False):
                for i in range(len(v_list)):
                    j=v_list[i]
                    self.v[j][k]=rng.random_sample(1)
        if self.N2==1:self.u=np.ones(self.u.shape)
            
                
    def _initialize(self,u_list,v_list,nodes):         
        
        rng=np.random.RandomState(self.rseed)   # Mersenne-Twister random number generator

        self._randomize_w(rng)  
        self._randomize_u_v(rng,u_list,v_list)        


    def _update_old_variables(self,u_list,v_list):
        for i in range(len(u_list)):
            for k in range(self.K):
                self.u_old[u_list[i]][k]=self.u[u_list[i]][k]   
        for i in range(len(v_list)):
            for k in range(self.K):self.v_old[v_list[i]][k]=self.v[v_list[i]][k]    
        for l in range(self.L):
            for k in range(self.K): 
                if(self.assortative==True):self.w_old[k][l]=self.w[k][l]    
                else:
                    for q in range(self.K):
                        self.w_old[k][q][l]=self.w[k][q][l] 


    def _update_optimal_parameters(self):
        self.u_f=np.copy(self.u)
        self.v_f=np.copy(self.v)
        self.w_f=np.copy(self.w)    

    # ----------    ----------  ----------  ----------  ----------  
    # ----------  Functions needed in the update_EM routine ----------  
    # ----------    ----------  ----------  ----------  ----------  

    def _update_U(self,A):
        if self.N2>1:
            Du=np.einsum('iq->q',self.v_old)
            if(self.assortative==False):
                w_k=np.einsum('kqa->kq',self.w_old)
                Z_uk=np.einsum('q,kq->k',Du,w_k)
                rho_ijka=np.einsum('jq,kqa->jka',self.v_old,self.w_old)
            else:
                w_k=np.einsum('ka->k',self.w_old)
                Z_uk=np.einsum('k,k->k',Du,w_k)
                rho_ijka=np.einsum('jk,ka->jka',self.v_old,self.w_old)
            
            rho_ijka=np.einsum('ik,jka->ijka',self.u,rho_ijka)

            Z_ija=np.einsum('ijka->ija',rho_ijka)
            Z_ijka=np.einsum('k,ija->ijka',Z_uk,Z_ija)

            non_zeros=Z_ijka>0.
            
            rho_ijka[non_zeros]/=Z_ijka[non_zeros]

            self.u=np.einsum('aij,ijka->ik',A,rho_ijka)
            low_values_indices = self.u < self.err_max  # Where values are low
            self.u[low_values_indices] = 0.  # All low values set to 0
            dist_u=np.amax(abs(self.u-self.u_old))  
            self.u_old=self.u
        else:dist_u=0.

        return dist_u   

    def _update_V(self,A):

        Dv=np.einsum('iq->q',self.u_old)
        if(self.assortative==False):
            w_k=np.einsum('qka->qk',self.w_old)
            Z_vk=np.einsum('q,qk->k',Dv,w_k)
            rho_jika=np.einsum('jq,qka->jka',self.u_old,self.w_old)

        else:   
            w_k=np.einsum('ka->k',self.w_old)
            Z_vk=np.einsum('k,k->k',Dv,w_k)
            rho_jika=np.einsum('jk,ka->jka',self.u_old,self.w_old)

        rho_jika=np.einsum('ik,jka->jika',self.v,rho_jika)
        
        Z_jia=np.einsum('jika->jia',rho_jika)
        Z_jika=np.einsum('k,jia->jika',Z_vk,Z_jia)
        non_zeros=Z_jika>0.

        rho_jika[non_zeros]/=Z_jika[non_zeros]

        self.v=np.einsum('aji,jika->ik',A,rho_jika)

        low_values_indices = self.v < self.err_max  # Where values are low
        self.v[low_values_indices] = 0.  # All low values set to 0

        dist_v=np.amax(abs(self.v-self.v_old))  
        self.v_old=self.v

        return dist_v       

    def _update_W(self,A):

        if(self.assortative==False):
            uk=np.einsum('ik->k',self.u)
            vk=np.einsum('ik->k',self.v)
            Z_kq=np.einsum('k,q->kq',uk,vk)
            #Z_kq=np.einsum('ik,jq->kq',self.u,self.v)
            Z_ija=np.einsum('jq,kqa->jka',self.v,self.w_old)
        else:
            uk=np.einsum('ik->k',self.u)
            vk=np.einsum('ik->k',self.v)
            Z_k=np.einsum('k,k->k',uk,vk)
            #Z_k=np.einsum('ik,jk->k',self.u,self.v)
            Z_ija=np.einsum('jk,ka->jka',self.v,self.w_old)
        
        Z_ija=np.einsum('ik,jka->ija',self.u,Z_ija)

        # if A.ndim==3:
        B=np.einsum('aij->ija',A)
        # else:B=A[np.newaxis,:,:]
        non_zeros=Z_ija>0.
        Z_ija[non_zeros]=B[non_zeros]/Z_ija[non_zeros]

        rho_ijkqa=np.einsum('ija,ik->jka',Z_ija,self.u)
        
        if(self.assortative==False):
            rho_ijkqa=np.einsum('jka,jq->kqa',rho_ijkqa,self.v)
            rho_ijkqa=np.einsum('kqa,kqa->kqa',rho_ijkqa,self.w_old)
            self.w=np.einsum('kqa,kq->kqa',rho_ijkqa,1./Z_kq)
        else: 
            rho_ijkqa=np.einsum('jka,jk->ka',rho_ijkqa,self.v)
            rho_ijkqa=np.einsum('ka,ka->ka',rho_ijkqa,self.w_old)
            self.w=np.einsum('ka,k->ka',rho_ijkqa,1./Z_k)
        
        low_values_indices = self.w < self.err_max  # Where values are low
        self.w[low_values_indices] = 0.  # All low values set to 0

        dist_w=np.amax(abs(self.w-self.w_old))  
        self.w_old=self.w

        return dist_w       

    
    def _update_em(self,B):
        if self.N2>1:
            d_u=self._update_U(B)
        else: d_u=0
        if(self.undirected==True):
            self.v=self.u
            self.v_old=self.v
            d_v=d_u
        else:   
            d_v=self._update_V(B)
        d_w=self._update_W(B)

        return d_u,d_v,d_w


    # --------------------------------------------------
    # Function needed to iterate
    # --------------------------------------------------                

    def _Likelihood(self,A):
        if(self.assortative==False):
            mu_ija=np.einsum('kql,jq->klj',self.w,self.v);
        else:   
            mu_ija=np.einsum('kl,jk->klj',self.w,self.v);
        mu_ija=np.einsum('ik,klj->lij',self.u,mu_ija);   
        l=-mu_ija.sum()
        non_zeros=A>0
        logM=np.log(mu_ija[non_zeros])
        Alog=A[non_zeros]*logM
        l+=Alog.sum()
        
        if(np.isnan(l)):
            print "Likelihood is NaN!!!!"
            sys.exit(1)
        else:return l           


    def _check_for_convergence(self,B,it,l2,coincide,convergence):
        if(it % 10 ==0):
            old_L=l2
            l2=self._Likelihood(B)  
            if(abs(l2-old_L)<self.tolerance): coincide+=1
            else: coincide=0
        if(coincide>self.decision):convergence=True 
        it+=1

        return it,l2,coincide,convergence   

    def cycle_over_realizations(self,B,u_list,v_list):
        maxL=-1000000000;
        nodes=np.arange(B.shape[1])
        if self.verbose:print('maxit=',self.maxit,self.rseed)
        for r in range(self.N_real):
                
            self._initialize(u_list,v_list,nodes)
            
            self._update_old_variables(u_list,v_list)

            # Convergence local variables
            coincide=0
            convergence=False
            it=0
            l2=self.inf
            #maxL=self.inf
            delta_u=delta_v=delta_w=self.inf

            tic=time.clock()
            # ------------------- Single step iteration update ------------------*/
            while(convergence==False and it<self.maxit):
                # Main EM update: updates membership and calculates max difference new vs old
                delta_u,delta_v,delta_w=self._update_em(B)
                it,l2,coincide,convergence=self._check_for_convergence(B,it,l2,coincide,convergence)
            if self.verbose:print("r=",r," Likelihood=",l2," iterations=",it,' time=',time.clock()-tic,'s')
            if(maxL<l2): 
                self._update_optimal_parameters()
                maxL=l2
            self.rseed+=1   
            self.maxL=l2
        # end cycle over realizations
        
        if self.verbose:print( "Final Likelihood=",maxL)

                
    
