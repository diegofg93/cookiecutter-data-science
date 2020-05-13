import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from  src.processing.errors import InvalidModelInputError



#.- HELPERS
def _define_variables(variables):
    # Check that variable names are passed in a list.
    # Can take None as value
    if not variables or isinstance(variables, list):
        variables = variables
    else:
        variables = [variables]
    return variables


def _find_numerical_variables(X, variables=None):
    # Find numerical variables in a data set or check that
    # the variables entered by the user are numerical.
    if not variables:
        variables = list(X.select_dtypes(include='number').columns)
    else:
        if len(X[variables].select_dtypes(exclude='number').columns) != 0:
            raise TypeError("Some of the variables are not numerical. Please cast them as numerical "
                            "before calling this transformer")
    return variables

def _is_dataframe(X):
    # checks if the input is a dataframe. Also creates a copy,
    # important not to transform the original dataset.
    if not isinstance(X, pd.DataFrame):
        raise TypeError("The data set should be a pandas dataframe")
    return X.copy()


def _check_input_matches_training_df(X, reference):
    # check that dataframe to transform has the same number of columns
    # that the dataframe used during fit method
    if X.shape[1] != reference:
        raise ValueError('The number of columns in this data set is different from that of the train set used during'
                         'the fit method')
    return None


def _check_contains_na(X, variables):
    if X[variables].isnull().values.any():
        raise ValueError('Some of the variables to trasnform contain missing values. Check and remove those '
                         'before using this transformer.')

        
        
class BaseNumericalTransformer(BaseEstimator, TransformerMixin):
    # shared set-up procedures across numerical transformers, i.e.,
    # variable transformers, discretisers, outlier handlers
    def fit(self, X, y=None):
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        return X

    def transform(self, X):
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns
        # than the dataframe used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X

class Winsorizer(BaseNumericalTransformer):
    """
    The Winsorizer() caps maximum and / or minimum values of a variable.
    
    The Winsorizer() works only with numerical variables. A list of variables can
    be indicated. Alternatively, the Winsorizer() will select all numerical
    variables in the train set.
    
    The Winsorizer() first calculates the capping values at the end of the
    distribution. The values are determined using 1) a Gaussian approximation,
    2) the inter-quantile range proximity rule or 3) percentiles.
    
    Gaussian limits:
        right tail: mean + 3* std
        left tail: mean - 3* std
        
    IQR limits:
        right tail: 75th quantile + 3* IQR
        left tail:  25th quantile - 3* IQR
    where IQR is the inter-quartile range: 75th quantile - 25th quantile.
    percentiles or quantiles:
        right tail: 95th percentile
        left tail:  5th percentile
    You can select how far out to cap the maximum or minimum values with the
    parameter 'fold'.
    If distribution='gaussian' fold gives the value to multiply the std.
    If distribution='skewed' fold is the value to multiply the IQR.
    If distribution='quantile', fold is the percentile on each tail that should
    be censored. For example, if fold=0.05, the limits will be the 5th and 95th
    percentiles. If fold=0.1, the limits will be the 10th and 90th percentiles.
    
    The transformer first finds the values at one or both tails of the distributions
    (fit).
    
    The transformer then caps the variables (transform).
    
    Parameters
    ----------
    
    distribution : str, default=gaussian
        Desired distribution. Can take 'gaussian', 'skewed' or 'quantiles'.
        gaussian: the transformer will find the maximum and / or minimum values to
        cap the variables using the Gaussian approximation.
        skewed: the transformer will find the boundaries using the IQR proximity rule.
        quantiles: the limits are given by the percentiles.
        
    tail : str, default=right
        Whether to cap outliers on the right, left or both tails of the distribution.
        Can take 'left', 'right' or 'both'.
        
    fold: int or float, default=3
        How far out to to place the capping values. The number that will multiply
        the std or IQR to calculate the capping values. Recommended values, 2 
        or 3 for the gaussian approximation, or 1.5 or 3 for the IQR proximity 
        rule.
        If distribution='quantile', then 'fold' indicates the percentile. So if
        fold=0.05, the limits will be the 95th and 5th percentiles.
        
    variables : list, default=None
        The list of variables for which the outliers will be capped. If None, 
        the transformer will find and select all numerical variables.
    """

    def __init__(self, distribution='gaussian', tail='right', fold=3, variables=None):

        if distribution not in ['gaussian', 'skewed', 'quantiles']:
            raise ValueError("distribution takes only values 'gaussian', 'skewed' or 'quantiles'")

        if tail not in ['right', 'left', 'both']:
            raise ValueError("tail takes only values 'right', 'left' or 'both'")

        if fold <= 0:
            raise ValueError("fold takes only positive numbers")

        self.distribution = distribution
        self.tail = tail
        self.fold = fold
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """ 
        Learns the values that should be used to replace outliers.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
        y : None
            y is not needed in this transformer. You can pass y or None.
        Attributes
        ----------
        right_tail_caps_: dictionary
            The dictionary containing the maximum values at which variables
            will be capped.
        left_tail_caps_ : dictionary
            The dictionary containing the minimum values at which variables
            will be capped.
        """
        # check input dataframe
        X = super().fit(X, y)

        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}

        # estimate the end values
        if self.tail in ['right', 'both']:
            if self.distribution == 'gaussian':
                self.right_tail_caps_ = (X[self.variables].mean() + self.fold * X[self.variables].std()).to_dict()

            elif self.distribution == 'skewed':
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.right_tail_caps_ = (X[self.variables].quantile(0.75) + (IQR * self.fold)).to_dict()

            elif self.distribution == 'quantiles':
                self.right_tail_caps_ = X[self.variables].quantile(1-self.fold).to_dict()

        if self.tail in ['left', 'both']:
            if self.distribution == 'gaussian':
                self.left_tail_caps_ = (X[self.variables].mean() - self.fold * X[self.variables].std()).to_dict()

            elif self.distribution == 'skewed':
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.left_tail_caps_ = (X[self.variables].quantile(0.25) - (IQR * self.fold)).to_dict()

            elif self.distribution == 'quantiles':
                self.left_tail_caps_ = X[self.variables].quantile(self.fold).to_dict()

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Caps the variable values, that is, censors outliers.
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.
        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """

        # check input dataframe an if class was fitted
        X = super().transform(X)

        for feature in self.right_tail_caps_.keys():
            X[feature] = np.where(X[feature] > self.right_tail_caps_[feature], self.right_tail_caps_[feature],
                                  X[feature])

        for feature in self.left_tail_caps_.keys():
            X[feature] = np.where(X[feature] < self.left_tail_caps_[feature], self.left_tail_caps_[feature], X[feature])

        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> 'CategoricalImputer':
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Rare label categorical encoder"""

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(
                self.encoder_dict_[feature]), X[feature], 'Rare')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """String to numbers categorical encoder."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = temp.groupby([var])['target'].mean().sort_values(
                ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if transformer introduces NaN
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {key: value for (key, value) in null_counts.items()
                     if value is True}
            raise InvalidModelInputError(
                f'Categorical encoder has introduced NaN when '
                f'transforming categorical variables: {vars_.keys()}')

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X

class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """Temporal variable calculator."""

    def __init__(self, variables=None, reference_variable=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variables = reference_variable

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variables] - X[feature]

        return X
