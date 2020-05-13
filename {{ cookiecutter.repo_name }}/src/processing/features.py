import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.processing.errors import InvalidModelInputError

import os
import logging
from typing import List, Tuple
from collections import namedtuple
import psutil
import re
import pandas as pd

_logger = logging.getLogger(__name__)


gb_memory = int(psutil.virtual_memory(
)._asdict().get("total") / (1024*(10**6)))


class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer."""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        # check that the values are non-negative for log transform
        if not (X[self.variables] > 0).all().all():
            vars_ = self.variables[(X[self.variables] <= 0).any()]
            raise InvalidModelInputError(
                f"Variables contain zero or negative values, "
                f"can't apply log for vars: {vars_}")

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X


def generate_additional_num_features(
        X: pd.DataFrame,
        num_features: List[str]
) -> pd.DataFrame:
    """
    Generate additional numerical features by applying mathematical operations on combinations of features
    :param X: Data
    :param num_features: Numerical features in data to use for combinations
    :return: Dataframe with new numerical features
    """
    if len(num_features) == 0:
        return X

    new_cols = []
    for i in range(len(num_features) - 1):
        for j in range(1, len(num_features)):
            col1, col2 = num_features[i], num_features[j]
            X["{}_add_{}".format(col1, col2)] = X[col1] + X[col2]
            X["{}_div_{}".format(col1, col2)] = X[col1] / X[col2]
            X["{}_mul_{}".format(col1, col2)] = X[col1] * X[col2]
            X["{}_min_{}".format(col1, col2)] = X[col1] - X[col2]

    for col in new_cols:
        X[col] = X[col].astype("float32")
    return X

def extract_time_features(df: pd.DataFrame, datetime_column):

    df = df.assign(**{"year":df.index.get_level_values(datetime_column).year,
                            "month":df.index.get_level_values(datetime_column).month
                           })

    time_periods = pd.DataFrame(df.index.get_level_values(datetime_column).unique()).sort_values(datetime_column).reset_index(drop=True).reset_index()
    time_periods = time_periods.rename(columns={"index": "time_period"})
    time_periods = time_periods.set_index(datetime_column)
    df = df.join(time_periods)

    return df

class AggregationOp(namedtuple("Op", ["c", "func", "alias"])):
    
    def expr(self):
        if callable(self.func):
            return self.func(self.c).alias(self.alias)
        else:
            return F.expr("{func}(`{c}`)".format
                (func = self.func, c = self.c)).alias(self.alias)
        
    def expr_window(self, window):
        if callable(self.func):
            return self.func(self.c).over(window).alias(self.alias)
        else:
            return F.expr("{func}(`{c}`)".format
                (func = self.func, c = self.c)).over(window).alias(self.alias)

def generate_rolling_features(df: pd.DataFrame,
                        selected_features: List[str],
                        spark_configuration: List,
                        ram_memory_use= False ):

    import findspark
    from pyspark import SparkContext, SQLContext, SparkConf
    from pyspark.sql import functions as F
    from pyspark.sql import SparkSession, Window
    
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"
    findspark.init("/opt/spark/")

    if not ram_memory_use:
        ram_memory_use = int(psutil.virtual_memory()._asdict().get("total") / (1024*(10**6)))

    _logger.info("Configuration spark session {}".format(spark_configuration))
    conf = SparkConf().setAll(spark_configuration)

    sc = SparkSession.builder.config(conf=conf).getOrCreate()
    print(sc)
    _logger.info("shape of DataFrame before rolling features {}".format(df.shape))

    sdf = sc.createDataFrame(df.reset_index())

    window_expanding = Window.partitionBy("user_id").orderBy(sdf["end_of_month"].asc())
    window_expanding_unbounded_2 = window_expanding.rowsBetween(-1, 0)
    window_expanding_unbounded_3 = window_expanding.rowsBetween(-2, 0)
    window_expanding_unbounded_4 = window_expanding.rowsBetween(-3, 0)

    aggregations = {"max": F.max, 
                    "min": F.min,
                    "avg": F.avg,
                    "sum": F.sum}

    pass_features = ['user_id', 'end_of_month', 'month', 'year', 'time_period']
    selected_columns_values = selected_features
    _logger.info("Getting new rolling features through {} variables".format(len(selected_columns_values)))
    ops = [AggregationOp(col, agg, "{}_rolling_{}".format(col, agg)) for col in selected_columns_values for agg in aggregations.keys()]
    ops_bounded_2 = [AggregationOp(col, agg, "{}_rolling_{}_2".format(col, agg)) for col in selected_columns_values for agg in aggregations.keys()]
    ops_bounded_3 = [AggregationOp(col, agg, "{}_rolling_{}_3".format(col, agg)) for col in selected_columns_values for agg in aggregations.keys()]
    ops_bounded_4 = [AggregationOp(col, agg, "{}_rolling_{}_4".format(col, agg)) for col in selected_columns_values for agg in aggregations.keys()]

    _logger.info("Calculating window period user")
    sdf = sdf.select( *[op.expr_window(window_expanding) for op in ops] + 
                 list(set(pass_features + selected_columns_values)) )

    _logger.info("Calculating window period user limit 2 periods")
    sdf_current_columns = sdf.columns
    #ventana deslizante 2 rows hacia atras
    sdf = sdf.select( *[op.expr_window(window_expanding_unbounded_2) for op in ops_bounded_2] + sdf_current_columns)
    sdf_current_columns = sdf.columns

    _logger.info("Calculating window period user limit 3 periods")
    #ventana deslizante 3 rows hacia atras
    sdf = sdf.select( *[op.expr_window(window_expanding_unbounded_3) for op in ops_bounded_3] + sdf_current_columns)
    sdf_current_columns = sdf.columns

    _logger.info("Calculating window period user limit 4 periods")
    #ventana deslizante 4 rows hacia atras
    sdf = sdf.select( *[op.expr_window(window_expanding_unbounded_4) for op in ops_bounded_4] + sdf_current_columns)
    sdf_current_columns = sdf.columns
    _logger.info("Total final columns in spark DataFrame: {}".format(len(sdf_current_columns)))

    return sdf

def get_additional_features(df, target, datetime_column, spark_configuration, 
                                columns="*",return_pandas= True, rolling_features= True):
    
    _logger.info("Creating time features ...")
    df = extract_time_features(df, datetime_column)
    original_columns = [col for col in df.columns if target not in col]
    _logger.info("Variables included in feature engineering {}".format(original_columns))
    _logger.info("Creating combinations with variables")
    df = generate_additional_num_features(df, original_columns)
    _logger.info("Replacing infinite values")
    df = df.replace(np.inf, 0)
    df = df.replace(-np.inf, 0)
    _logger.info("Filtering only useful variables after creater combination")
    final_primitive_columns = list(set([re.sub(r'_rolling.*', "" ,x) for x in columns]))
    df = df[list(set(original_columns).union(set(final_primitive_columns)))]
    _logger.info("Creating time related features for: {}".format(list(df)))
    extended_columns = [col for col in df.columns if target not in col]


    if rolling_features:
        _logger.info("Using spark for generating rolling features ...")
        df = generate_rolling_features(df=df, selected_features=extended_columns, spark_configuration=spark_configuration)

    if return_pandas:
        _logger.info("Transforming spark DataFrame into pandas DataFrame ...")
        df = df.select(["user_id", "end_of_month"] + columns)
        df = df.toPandas()
        _logger.info("Setting user_id and end_of_month as index ...")
        df = df.sort_values(["user_id", "end_of_month"]).set_index(["user_id", "end_of_month"])
        return df

    else:
        return df