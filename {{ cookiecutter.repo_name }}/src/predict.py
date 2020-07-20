import sys
import os
#add to path to search interpreter parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import pandas as pd

from src.processing.data_management import load_pipeline
from src.config import config, logging_config
from src.processing.validation import validate_inputs
from src import __version__ as _version

_logger = logging_config.get_logger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}{_version}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _price_pipe.predict(validated_data[config.FEATURES])
    output = np.exp(prediction)

    results = {'predictions': output, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results