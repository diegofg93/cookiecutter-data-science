import sys
import os
from datetime import datetime
# add to path to search interpreter parent directory
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))


import numpy as np
from sklearn.model_selection import train_test_split

from src import pipeline
from src.processing.data_management import (
    load_dataset, save_pipeline)
from src.config import config, logging_config
from src import __version__ as _version
import click
from pandas_profiling import ProfileReport


_logger = logging_config.get_logger(__name__)


@click.command()
@click.option('--experiment_name', default=False, help='If true, a pdf report of the data is created')
@click.option('--data_profiling', default=True, help='If true, a pdf report of the data is created')
def run_training(experiment_name,
                 data_profiling) -> None:
    """Train the model."""
    _logger.info(f'Working on: {os.getcwd()}')
    if experiment_name == False:
        experiment_name = datetime.now().strftime("model_experiment_%Y%m%d_%H%M%S")
        _logger.info(
            "The run training name was fixed in {}".format(experiment_name))

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    _logger.info("The dataset contains {} rows and {} columns".format(
        data.shape[0], data.shape[1]))
    _logger.info("Dataset info: \n{}". format(
        data.describe(percentiles=[], include="all").T.to_string()))

    model_subfloder = config.TRAINED_MODEL_DIR/experiment_name
    _logger.info("Creating model folder in {}".format(str(model_subfloder)))

    if data_profiling:
        _logger.info("Creating a data report for data training")
        profile = ProfileReport(data, title=experiment_name, explorative=True)
        profile.to_file(config.REPORT_DIR / "data_train_report.html")
        _logger.info("A report in html was saved in {}".format(config.REPORT_DIR))

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
