import logging
import os

from src.config import config
from src.config import logging_config


# Configure logger for use in package
logger = logging_config.get_logger(__name__)


with open(os.path.join(config.PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
