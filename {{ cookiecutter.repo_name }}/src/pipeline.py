import importlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.processing import preprocessors as pp
from src.processing import features
from src.config import config

import logging


_logger = logging.getLogger(__name__)

Algorithm = getattr(
    importlib.import_module(config.cfg.train.algorithm_module), config.cfg.train.algorithm_name
)

price_pipe = Pipeline(
    [
        ('categorical_imputer',
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),
        ('numerical_inputer',
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),
        ('temporal_variable',
            pp.TemporalVariableEstimator(
                variables=config.TEMPORAL_VARS,
                reference_variable=config.DROP_FEATURES)),
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.01,
                variables=config.CATEGORICAL_VARS)),
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('log_transformer',
            features.LogTransformer(variables=config.NUMERICALS_LOG_VARS)),
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),
        ('scaler', MinMaxScaler()),
        ('Linear_model', Algorithm(**config.cfg.train.algorithm_params))
    ]
)