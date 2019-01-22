"""
Factor model cross validation object
"""
import numpy as np
from numpy.random import RandomState
import numbers

class cv_fold_fm:

    def __init__(self, holdout_portion, random_state):
            if not isinstance(holdout_portion, numbers.Real):
                raise ValueError('The number of folds must be a number type. '
                                 '%s of type %s was passed.'
                                 % (holdout_portion, type(holdout_portion)))
            holdout_portion = float(holdout_portion)

            if (holdout_portion <= 0.0) or (holdout_portion>=1.):
                raise ValueError(
                    "k-fold cross-validation requires a valid"
                    " train/test split by setting n_splits>0.0 and n_splits<1.0,"
                    " got n_splits={0}.".format(holdout_portion))

            self.holdout_portion = holdout_portion
            self.random_state = random_state

            self.prng=np.random.RandomState(seed=self.random_state)


    def split(self,df,stratified=False):

        N=len(df)
        
        permuted=self.prng.permutation(N)
        test=permuted[:int(self.holdout_portion * N)]
        train=permuted[int(self.holdout_portion * N):]

        test_idx = np.zeros(N, dtype=bool)
        test_idx[test] = True

        test_df = df[test_idx]
        train_df = df[~test_idx]

        return test_df,train_df