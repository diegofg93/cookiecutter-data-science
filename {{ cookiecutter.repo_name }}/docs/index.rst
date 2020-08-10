.. {{ cookiecutter.project_name }} documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

{{ cookiecutter.project_name }} documentation!
==============================================

Contents:

.. toctree::
   :maxdepth: 2

   getting-started
   commands



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Pipeline
===================
.. automodule:: src.pipeline
   :members:
Train the model
===================
.. automodule:: src.predict
   :members:
Predict
===================
.. automodule:: src.train_pipeline
   :members:

Models
===================
.. automodule:: src.models
   :members:

clustering_training
--------------------
.. automodule:: src.models.clustering_training
   :members:

utils
------
.. automodule:: src.models.utils_model
   :members:


Processing
===========
.. automodule:: src.processing
   :members:

Data management
---------------
.. automodule:: src.processing.data_management
   :members:

Preprocessors
--------------
.. automodule:: src.processing.preprocessors
   :members:

features
--------
.. automodule:: src.processing.features
   :members:

errors
------
.. automodule:: src.processing.errors
   :members:

validation
----------
.. automodule:: src.processing.validation
   :members:

