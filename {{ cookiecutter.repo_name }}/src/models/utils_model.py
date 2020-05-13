import pandas as pd 
import numpy as np

class ModifiedTimeSeriesSplit(_BaseKFold):
    
    def __init__(self, name_group, n_splits=5):
            super().__init__(n_splits, shuffle=False, random_state=None)
            self.name_group = name_group
    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        
        X, y, groups = indexable(X, y, groups)
        groups= X[self.name_group]
        n_samples = _num_samples(X)
        n_splits = self.n_splits

        indices = np.arange(n_samples)
        test_starts = range(groups.max() - n_splits + 1, groups.max() + 1)
      
        for test_start in test_starts:
            yield (indices[0:len(X[X[self.name_group] < test_start])],
                  indices[len(X[X[self.name_group] < test_start]):len(X[X.time_period <= test_start])] )