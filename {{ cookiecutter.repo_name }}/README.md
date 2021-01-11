{{cookiecutter.project_name}}
==============================

{{cookiecutter.description}}

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands for install package as developer or create environment
    ├── README.md          <- The top-level README for developers using this project.
    ├── datasets
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details. Creating using MAKEFILE inside owns folder
    │
    ├── ml_api             <- An API for serving the model, in very early stage.
    │
    │── logs               <- Folder to save logs generated.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`. Furthermore you can find some examples notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── trained_models     <- Trained and serialized models, model predictions, or model summaries
    │
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`. You can install from MAKEFILE running make requirements
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported, you can install package as editable with: make install_editable 
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │   
    │   ├── pipeline.py    <- Contains the configuration of the model pipeline, like a sklearn pipeline
    │   ├── train_pipeline.py <- Train and generate the model
    │   ├── predict.py     <- Use the trained model for predicting data
    │   │
    │   ├── config
    │   │   ├── config.py   <- Config with basic information as dataset directory.
    │   │   ├── external_configs.py <- Custom information of the project like databases, or training variables.
    │   │   └── logging_config.py   <- Configuration of the logger in the package.
    │   │
    │   ├── processing      <- Module that contains data function.
    │   │   └── data_management.py <- Contains functions for load data.
    │   │   ├── errors.py          <- You can generate custom erros in this module.
    │   │   ├── features.py        <- Module for generating aditional features, contains some advances examples in spark.
    │   │   ├── preprocessors.py   <- Custom preprocessor often plugin in scikit learn pipeline.
    │   │   └── validation.py      <- Checker functions for data.
    │   │
    │   │
    │   ├── models         <- Scripts to train models.
    │   │   ├── clustering_training.py  <- Class implemented for automatic clustering training.
    │   │   └── utils_models.py         <- Functions that could be use in several training models.
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations.
    │       └── visualize.py
    │
    │── ml_api        <- Folder with a skeleton for deploy your app with flask.
    │   ├── __init__.py    <- Makes ml_api a Python module.
    │   ├── api            <- Folder with the flask app.
    |   |     ├── __init__.py    <- Makes api a Python module.
    |   |     ├── app.py         <- Create the flask app.
    |   |     ├── config.py      <- Configure the logger for the flask app.
    |   |     ├── controller.py  <- Routes in the flask app.
    |   |     ├── validation.py  <- Schema validation for the data.
    │   ├── tests      <- tests for the flask api..
    |   |     ├── conftest.py           <- Configuration test.
    |   |     ├── test_controller.py    <- Test check api.
    |   |     ├── test_validation.py    <- Test for validation input data.
    │   ├── run.py         <- Run the flask application.
    │   ├── requirements.txt      <- The requirements for run the flask aplication.
    │   └── VERSION            <- Version of the api.
    │
    │── tests <- folder that contains the test of the code
    │   ├── test_predict.py    <- Tests for checking the predictions.
    │
    │── test_environment.py <- Check environments configurations.
    │
    │── .env        <- You can add environment variables and load with python-dotenv, (NEVER COMMIT).
    │
    │── .gitignore  <- Configuration of files that can never be commited.
    │
    │── MAKEFILE contains a bunchs of command for an easy installation 
    │
    │── MANIFEST.in When building a source distribution for your package, by default only a minimal set of files are included: https://packaging.python.org/guides/using-manifest-in/
    │
    │── VERSION     <- Contains the reference of the version
    │
    └── tox.ini     <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
