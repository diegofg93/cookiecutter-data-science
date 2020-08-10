import warnings
from collections import Counter
import os
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from collections.abc import Mapping, Sequence, Iterable
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas_profiling import ProfileReport
from sklearn.utils.validation import check_is_fitted
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import sklearn.cluster as cluster
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.processing import preprocessors as pp
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import umap
from yellowbrick.cluster import SilhouetteVisualizer
import plotly.express as px
import plotly.io as pio
from src.visualization import visualize
from src.processing import data_management
from src.processing import preprocessors as pp
from src.config import config
import logging

logger = logging.getLogger(__name__)


def create_param_grid(winsorizer_columns, winsorizer_distribution="quantiles",
                      winsorizer_tail="right", winsorizer_fold=0.01, min_clusters=5, max_clusters=12):

    winsorizer = pp.Winsorizer(distribution=winsorizer_distribution,
                               tail=winsorizer_tail,
                               fold=winsorizer_fold,
                               variables=winsorizer_columns
                               )

    preprocessors = [preprocessing.PowerTransformer()]
    kmeans_clusters = [cluster.KMeans(n_clusters=i, init="k-means++", n_init=10,
                                      max_iter=1000, random_state=42) for i in range(min_clusters, max_clusters)]
    #dbscan_clusters=[cluster.DBSCAN(min_samples=i) for i in range(df.shape[1], df.shape[1] * 2)]
    #birch_clusters=[cluster.Birch(n_clusters=i) for i in range(5,7)]

    clustering_param_grid = {
        "transform_inputs__numerical_preprocessing__preprocessor": [None, winsorizer],
        "transform_inputs__numerical_preprocessing__transformer": preprocessors,
        "transform_inputs__numerical_preprocessing__reduce_dim": [None, PCA(.9)],
        "clustering": kmeans_clusters,  # +birch_clusters+dbscan_clusters,
    }

    return clustering_param_grid


def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    X_transformed = estimator.named_steps['transform_inputs'].transform(X)
    try:
        cluster_labels = estimator.labels_
    except Exception as e:
      # print(e,estimator)
        cluster_labels = estimator.predict(X)
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return metrics.silhouette_score(X_transformed, cluster_labels)


class ClusteringTraining:
    """
    A class to help with the automatic training of the cluster and 
    its respective report of the training results.
    ...

    Attributes
    ----------
    param_grid : list
        A list with differents options for searching in your grid search

    numerical_columns : str
        A list with the names of the numerical columns

    categorical_columns : list
        A list with the names of the categorical columns

    data : pandas.core.frame.DataFrame
        A dataframe with the training data

    grid_search_result : sklearn.model_selection._search.GridSearchCV
        A grid search fitted

    output_directory : str
        The directory where all the outputs are going to be saved

    profile : pandas_profiling.ProfileReport
        A profile report of the training data made by pandas-profilling library

    pipeline : sklearn.pipeline.Pipeline
        The pipeline created by the method configure_clustering_preprocessing_pipeline,
        another pipeline could be assign, but it must have the final steps equal than in 
        the method ("transform_inputs", "clustering")

    experiments_result_ : pandas.core.frame.DataFrame
        A pandas dataframe with the results of the grid search fitted

    best_estimator_ : sklearn.pipeline.Pipeline
        a pipeline with the best model trained according to the chosen metric

    cluster_labels_predicted_ : numpy.ndarray
        The predicted cluster using best_estimator_

    df_full : pandas.core.frame.DataFrame
        The data plus a new columns with the predicted label assigned named cluster

    data_preprocessed : pandas.core.frame.DataFrame
        A pandas dataframe with the data transformed by transform_inputs
        step pipeline

    number_per_cluster : list
        Contains the numbers per each cluster in df_full

    viz_matrix = viz_matrix : pandas.core.frame.DataFrame
        A pandas dataframe with the components result of fit a embeddings model

    viz_matrix_extra : pandas.core.frame.DataFrame
        viz_matrix plus the original data

    umpap_fig : plotly.graph_objs._figure.Figure
        A visual representation of the data with a dimensionaloty reduction technique
    
    Methods
    -------
    make_data_training_report(name_report=data_training_analysis)
        Create a html report with the training data

    Examples
    --------
    >>> from sklearn import cluster, preprocessing, imputer, datasets
    >>> from src.config import config

    >>> X, y = datasets.make_blobs(n_samples=100, centers=3, n_features=4)
    >>> df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], z=X[:,2], g=X[:,3],label=y))
    >>> param_grid = create_param_grid(
    ...    winsorizer_columns=["x", "y],
    ...    min_clusters=5,
    ...    max_clusters=20)

    >>> clustering_training = ClusteringTraining(data=df,
    ...     numerical_columns=["x", "y", "z", "g"]
    ...     categorical_columns=[],
    ...     param_grid=param_grid,
    ...     output_directory=config.TRAINED_MODEL_DIR
    ...                                          )
    >>> clustering_training.make_data_training_report()

    >>> clustering_training.configure_clustering_preprocessing_pipeline(
    ...     numerical_preprocessor="passthrough",
    ...     numerical_imputer=impute.SimpleImputer(),
    ...     numerical_transformer="passthrough",
    ...     categorical_imputer=impute.SimpleImputer(strategy='most_frequent'),
    ...     categorical_transformer=preprocessing.OneHotEncoder(
    ...         categories='auto', sparse=False, handle_unknown='ignore'),
    ...     reduced_dimension_algorithm="passthrough",
    ...     cluster_algorithm="passthrough"
    ...  )

    >>> clustering_training.training_grid_search(n_jobs=-2, 
    ...    scorer=clusrering_training.cv_silhouette_scorer)

    >>> clustering_training.create_cluster_analysis_report()

    >>> clustering_training.create_umap_cluster_representation(
    ...    neighbors=20, n_components=3, min_dist=.5,
    ...    hover_name=df.index,
    ...    hover_data=clustering_training.categorical_columns+clustering_training.numerical_columns)

    """    

    def _check_scikit_pipeline(self):
        print(self.pipeline)

    def __init__(self, data: pd.core.frame.DataFrame, numerical_columns: List[str], categorical_columns: List[str], param_grid:dict, output_directory: str):
        """
        Args:
            data (pd.core.frame.DataFrame): 
                A pandas dataframe with the data
            numerical_columns (List[str]): A list of numerical columns
            categorical_columns (List[str]): A list of categorical columns
            param_grid (dict): A dict with param grid, you can use the function
            create_param_grid
            output_directory (str): The directory when the results of the traingin are
            going to be saved

        Raises:
            TypeError: You must pass a consistency param grid, check:
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        """        
        self.param_grid = param_grid
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.data = data[numerical_columns+categorical_columns]
        self.grid_search_result = None
        self.output_directory=output_directory

        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))

        if len(self.categorical_columns) > 0:
            warnings.warn(
                "Be careful when working with categorical variables in clustering. "
                "most algorithms use distance measurements that are only effective with continuous variables.."
                "For more information: http://www.cs.ust.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf",
                UserWarning,
                stacklevel=2,
            )

        logger.info("Training clustering dataset instantiated with succes!")
        logger.info("Dataset contains {} rows and {} columns".format(
            self.data.shape[0], self.data.shape[1]))
        logger.debug("{} Numerical columns:\n {}".format(len(self.numerical_columns),
                                                         str(self.numerical_columns)))
        logger.info("{} Categorical columns: {}".format(len(self.categorical_columns),
                                                        str(self.categorical_columns)))
        logger.info("descriptive analysis:\n{}".format(
            self.data.describe().T.to_string()))
        logger.info("Param grid specified:\n{}".format(str(self.param_grid)))

        logger.info("Creating the directory indicated ...")
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

    def make_data_training_report(self, name_report: str="data_training_analysis"):
        """
        Create a html report with the training data

        Args:
            name_report (str, optional): [description]. Defaults to "data_training_analysis".
        """        
        
        logger.info("Creating html report about input data ...")
        self.profile=ProfileReport(self.data.reset_index(), title="Pandas Profiling Report")
        self.profile.to_file(self.output_directory+"/{}.html".format(name_report))
        logger.info("The report was saved in {}...".format(self.output_directory+"{}.html".format(name_report)))

    def configure_clustering_preprocessing_pipeline(self,
                                                    numerical_preprocessor="passthrough",
                                                    numerical_imputer="passthrough",
                                                    numerical_transformer="passthrough",
                                                    categorical_imputer="passthrough",
                                                    categorical_transformer="passthrough",
                                                    reduced_dimension_algorithm="passthrough",
                                                    cluster_algorithm="passthrough"
                                                    ):
        """
        This method create the base pipeline for training several clustering with a bunch of
        params. You have to keep the argument passthrough in those parameter contained in param
        grid.

        Args:
            numerical_preprocessor (str, optional): Some preparation with data like winsorizer. 
            Defaults to "passthrough".
            numerical_imputer (str, optional): Imputing numerical missing values . 
            Defaults to "passthrough".
            numerical_transformer (str, optional): Some transformer like standard scaler. 
            Defaults to "passthrough".
            categorical_imputer (str, optional): Imputing categorical missing values. 
            Defaults to "passthrough".
            categorical_transformer (str, optional): Some transformer like One hot encoder. 
            Defaults to "passthrough".
            reduced_dimension_algorithm (str, optional): Some algorithm like PCA. 
            Defaults to "passthrough".
            cluster_algorithm (str, optional): Cluster Algorithm. 
            Defaults to "passthrough".
        """          

        numerical_step = Pipeline(steps=[
            ("preprocessor", numerical_preprocessor),
            ("imputer", numerical_imputer),
            ("transformer", numerical_transformer),
            ("reduce_dim", reduced_dimension_algorithm)
        ])

        cat_step = Pipeline(steps=[
            ('imputer', categorical_imputer),
            ('transformer', categorical_transformer)
        ])

        preprocessing_variables_pipeline = ColumnTransformer(transformers=[
            ('numerical_preprocessing', numerical_step, self.numerical_columns),
            ('categorical_preprocessing', cat_step, self.categorical_columns)
        ],
            remainder='drop',
            verbose=True)

        pipeline = Pipeline([
            ("transform_inputs", preprocessing_variables_pipeline),
            ("clustering", cluster_algorithm)
        ])

        self.pipeline = pipeline
        logger.info("configured pipeline:\n {}".format(str(self.pipeline)))

    def training_grid_search(self,
                             scorer,
                             n_jobs=-2,
                             save_experiments_results=True
                             ):
        """
        This method train the Grid search.
        Args:
            scorer ([type]): Function to evaluate tha cluster assigned
            n_jobs (int, optional): Number of cores used in the process. Defaults to -2.
            save_experiments_results (bool, optional): Save the trained results in a csv. Defaults to True.
        """        

        logger.info("Training the models")
        gs = GridSearchCV(estimator=self.pipeline,
                          param_grid=self.param_grid,
                          scoring=scorer,
                          cv=[(slice(None), slice(None))],
                          n_jobs=n_jobs)

        self.grid_search_result = gs.fit(self.data)
        self.experiments_result_ = pd.DataFrame(
            self.grid_search_result.cv_results_).sort_values("rank_test_score")
        self.best_estimator_ = self.grid_search_result.best_estimator_
        self.cluster_labels_predicted_ = self.grid_search_result.best_estimator_.predict(
            self.data)
        self.df_full = self.data.copy()
        self.df_full["cluster"] = self.cluster_labels_predicted_
        self.data_preprocessed = pd.DataFrame(
            self.best_estimator_.named_steps['transform_inputs'].transform(self.data))
        self.number_per_cluster = sorted(
            Counter(self.cluster_labels_predicted_).items())
        logger.debug("Max score: {}".format(
            self.experiments_result_["mean_test_score"].max()))
        logger.debug("Sample per cluster: {}".format(
            str(self.number_per_cluster)))
        logger.debug("Best pipeline:\n {}".format(
            str(self.best_estimator_.get_params())))

        logger.info("Save pipeline with best model...")
        data_management.save_pipeline(
            pipeline_to_persist=self.best_estimator_, 
            folder=self.output_directory,
            name_pickle="clustering_pipeline")

        if save_experiments_results:
            logger.info("Saving result experiment in {}".format(self.output_directory))
            self.experiments_result_.to_csv(self.output_directory+"/grid_search.csv")

            

    def _check_model(self):

        if self.grid_search_result == None:
            raise Exception(" Call 'training_grid_search' with "
                            "appropriate arguments before using this method.")

        check_is_fitted(self.grid_search_result, msg="The model is not fitted yet. Call 'training_grid_search' with "
                        "appropriate arguments before using this method.")

    # visualization functions
    def create_cluster_sumary(self):
        logger.info("Creating cluster variables summary")
        self._check_model()
        self.cluster_summary = visualize.comparare_clusters(self.data,
                                                            cluster_labels=self.cluster_labels_predicted_,
                                                            categorical_columns_names=self.categorical_columns,
                                                            cluster_ids=np.unique(
                                                                self.cluster_labels_predicted_)
                                                            ).style.background_gradient(cmap='Blues', axis=1)

    def create_descriptive_analysis_variables(self):
        logger.info("Creating cluster variables summary detailed")
        self._check_model()
        self.descriptive_analysis_variables = self.df_full.groupby(
            "cluster").describe().T.style.background_gradient(cmap='Blues', axis=1)

    def shiloutte_score_plot(self, directory):
        self._check_model()
        plt.figure(figsize=(10, 10))
        visualizer = SilhouetteVisualizer(
            self.best_estimator_.named_steps['clustering'], colors='yellowbrick', is_fitted=True)
        visualizer.fit(self.data_preprocessed)
        visualizer.show(directory+"/shiloutte_score.png")
        visualizer.finalize()
        plt.close()

    def create_umap_cluster_representation(self, neighbors=20, n_components=3, min_dist=.5, metric="euclidean",plot_name="index", **kwargs):
        self._check_model()
        #neighbors = data.shape[1] - 1
        names = ['component_1', 'component_2', 'component_3']
        logger.info("Training UMAP ...")
        manifold = umap.UMAP(n_neighbors=neighbors, n_components=n_components, min_dist=min_dist, metric=metric)
        embeddings = manifold.fit_transform(self.data_preprocessed)
        viz_matrix = pd.DataFrame(embeddings)
        viz_matrix.rename({i: names[i] for i in range(n_components)}, axis=1, inplace=True)

        viz_matrix['cluster'] = self.cluster_labels_predicted_
        viz_matrix = viz_matrix.set_index(self.data.index)
        viz_matrix_extra = self.data.join(viz_matrix)

        self.viz_matrix = viz_matrix
        self.viz_matrix_extra = viz_matrix_extra

        logger.info("Building plot ...")
        self.umpap_fig = px.scatter_3d(self.viz_matrix_extra,
                            x='component_1',
                            y='component_2',
                            z='component_3',
                            color="cluster",
                            opacity=0.5,
                            **kwargs)
        self.umpap_fig.update_traces(marker=dict(size=2))
        # fig.show()
        logger.info("Saving in {}...".format(self.output_directory+'/{}.html'.format(plot_name)))
        pio.write_html(self.umpap_fig, file=self.output_directory+'/{}.html'.format(plot_name), auto_open=False)

    def create_cluster_analysis_report(self):
        logger.info("Create visualization report in:{}".format(self.output_directory))
        self._check_model()

        self.create_cluster_sumary()
        self.cluster_summary.to_excel(self.output_directory+"/cluster_summary.xls",
                                      sheet_name='Sheet1',
                                      index=True,
                                      engine='openpyxl')
        logger.info("cluster_summary.xls saved")
        with open(self.output_directory+"/cluster_summary.html", "w") as file:
            file.write(self.cluster_summary.render())
        logger.info("cluster_summary.html saved")

        self.create_descriptive_analysis_variables()
        self.descriptive_analysis_variables.to_excel(self.output_directory+"/descriptive_analysis_variables.xls",
                                                     sheet_name='Sheet1',
                                                     index=True,
                                                     engine='openpyxl')
        logger.info("descriptive_analysis_variables.xls saved")
        with open(self.output_directory+"/descriptive_analysis_variables.html", "w") as file:
            file.write(self.descriptive_analysis_variables.render())
        logger.info("descriptive_analysis_variables.html saved")

        logger.info("Creating cluster variables distribution plot")
        axes = visualize.comparare_clusters(self.data,
                                            cluster_labels=self.cluster_labels_predicted_,
                                            categorical_columns_names=self.categorical_columns,
                                            cluster_ids=np.unique(self.cluster_labels_predicted_)).T.plot.bar(subplots=True, figsize=(25, 60), sharex=False, layout=(-1, 2))
        plt.savefig(
            self.output_directory+"/cluster_variables_distribution.png")
        plt.close()
        logger.info("cluster_variables_distribution.png saved")

        self.shiloutte_score_plot(directory=self.output_directory)

        # pdf reporting with variables
        logger.info("Create pdf reporting for every variable in each cluster")
        pdf = PdfPages(
            self.output_directory+"/distribution_variables_groups.pdf")
        for var in self.data[self.numerical_columns]:
            fig = visualize.analyse_continous(
                self.data, var, cluster_labels=self.cluster_labels_predicted_)
            pdf.savefig(fig)
            plt.close("all")
        pdf.close()
        logger.info("distribution_variables_groups.pdf saved")
