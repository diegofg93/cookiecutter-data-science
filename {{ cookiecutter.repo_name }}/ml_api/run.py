##############################################################################
# That's the entrypoint for deploying your machine learning app as a REST api 
# if you want.
# For that you have to install the requirements in that package
# from the parent directory of the package:
#
#    $ pip install -r ml_api/requirements.txt
#
# That it will install your ml app and all its dependencies besides
# flask.
#
# You can check if everything is ok using the tests moving to ml_api and 
# testing with
#
#    $pytest tests
# 
# Then move to ml_api directory and introduce:
#
#    $python run.py
#
###############################################################################
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
#this is basically telling flask what the entry point is to start up our API
os.environ["FLASK_APP"] = "run.py" 

from api.app import create_app
from api.config import DevelopmentConfig

application = create_app(
    config_object=DevelopmentConfig)

if __name__ == '__main__':
    application.run()
