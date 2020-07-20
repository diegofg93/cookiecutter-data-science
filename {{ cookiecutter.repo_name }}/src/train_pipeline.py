import sys
import os
#add to path to search interpreter parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))


import numpy as np
from sklearn.model_selection import train_test_split

from src import pipeline
from src.processing.data_management import (
    load_dataset, save_pipeline)
from src.config import config, logging_config
from src import __version__ as _version


_logger = logging_config.get_logger(__name__)

def run_training() -> None:
    """Train the model."""
    _logger.info(f'Working on: {os.getcwd()}')
    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.1,
        random_state=0)  # we are setting the seed here

    # transform the target
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    pipeline.price_pipe.fit(X_train[config.FEATURES],
                            y_train)

    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.price_pipe)

    _logger.info(f'Logs saved on: {config.LOG_DIR}')

if __name__ == '__main__':
    run_training()