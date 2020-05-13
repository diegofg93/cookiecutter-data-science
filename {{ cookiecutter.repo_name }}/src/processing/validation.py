from src.config import config

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.NUMERICAL_NA_NOT_ALLOWED)

    # check for categorical variables with NA not seen during training
    if input_data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.CATEGORICAL_NA_NOT_ALLOWED)

    # check for values <= 0 for the log transformed variables
    if (input_data[config.NUMERICALS_LOG_VARS] <= 0).any().any():
        vars_with_neg_values = config.NUMERICALS_LOG_VARS[
            (input_data[config.NUMERICALS_LOG_VARS] <= 0).any()]
        validated_data = validated_data[
            validated_data[vars_with_neg_values] > 0]

    return validated_data


def duplicate_values_col(df, threshold_cardinality = 0.85, exclude=[]):
    '''Nos dice cuantas veces se repite el valor más comun y cuanto el menos común'''
    columnas_cardinalidad = []
    columnas_altisima_cardinalidad = []
    n_records = len(df)
    
    column_names = []
    more_common_values = []
    less_common_values = []
    more_common_names = []
    less_common_names = []
    unique_values_column = []

    
    for columna in df.drop(exclude, 1):
        n_por_valor = df[columna].value_counts()
        mas_comun = (n_por_valor.iloc[0] / (1.0*n_records)) * 100
        menos_comun = (n_por_valor.iloc[-1] / (1.0*n_records)) * 100
        nombre_comun = n_por_valor.index[0]
        nombre_menos_comun = n_por_valor.index[-1]
        unique_values = len(df[columna].unique())
        
        column_names.append(df[columna].name)
        more_common_values.append(mas_comun)
        less_common_values.append(menos_comun)
        more_common_names.append(nombre_comun)
        less_common_names.append(nombre_menos_comun)
        unique_values_column.append(unique_values)        
        
        if (mas_comun / (1.0*n_records)) > 0.5 and  (mas_comun / (1.0*n_records)) < threshold_cardinality:
            columnas_cardinalidad.append(df[columna].name)
        elif (mas_comun / (1.0*n_records)) >= threshold_cardinality:
            columnas_altisima_cardinalidad.append(df[columna].name)
        else:
            pass
            
        
    info_columns_dict = {'column_name': column_names,
                         'unique_values': unique_values_column,
                        'more_common_value_%': more_common_values,
                        'less_common_value_%': less_common_values,
                        'more_common_name': more_common_names,
                        'less_common_names': less_common_names}
    
    info_columns = pd.DataFrame(info_columns_dict)
    info_columns = info_columns[['column_name', 'unique_values' ,'more_common_value_%', 
                                'more_common_name', 'less_common_value_%', 'less_common_names']]
            
    return columnas_cardinalidad, columnas_altisima_cardinalidad, info_columns

def outliers_col(df, z_score_level_outlier=3):
    
    list_column_names = list()
    list_n_outliers = list()
    list_min_value_outliers = list()
    list_max_value_outliers = list()
    list_type_columns = list()
    list_medium_value_col = list()
    list_median_value_col = list()
    
    
    for columna in df:
        if df[columna].dtype in [int, float, np.int64, np.float64]:
            outliers_df = df[np.abs(stats.zscore(df[columna])) > z_score_level_outlier]
            n_outliers = len(outliers_df)
            if n_outliers > 0:
                
                list_column_names.append(str(df[columna].name))
                list_n_outliers.append(n_outliers)
                list_max_value_outliers.append(round(outliers_df[columna].max(),2))
                list_min_value_outliers.append(round(outliers_df[columna].min(),2))
                list_medium_value_col.append(df[columna].mean())
                list_median_value_col.append(df[columna].median())
            
        else:
            print(columna, "excluded")
    dict_values_outliers = {"name_column": list_column_names,
                            "number_outliers": list_n_outliers,
                            "max_value_outliers": list_max_value_outliers,
                            "min_value_outliers": list_min_value_outliers,
                            "mean_column": list_medium_value_col,
                            "median_column": list_median_value_col
                                        
                                       }
    dataframe = pd.DataFrame.from_dict(dict_values_outliers)
    dataframe = dataframe[["name_column", "number_outliers", "max_value_outliers", "min_value_outliers", "mean_column", "median_column"]]
    return dataframe