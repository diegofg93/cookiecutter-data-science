import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.regressor.alphas import AlphaSelection
import yellowbrick as yb
from typing import List, Dict
import warnings
import sklearn
import matplotlib


# .- EDA Functions ---------------------------------------------------
def analyse_continous(df, var, cluster_labels):
    df = df.copy()

    # selecting number of bins
    selected_bins = 30
    unique_values = len(df[var].unique())
    if unique_values < 30:
        selected_bins = unique_values

    plt.figure()
    fig = df[var].hist(by=cluster_labels,
                       layout=(1, len(np.unique(cluster_labels))),
                       bins=selected_bins,
                       figsize=(20, 10), stacked=True)
    plt.ylabel('Number of Customers')
    plt.xlabel(var)
    plt.title(var)

# .- regression functions ---------------------------------------------------


def plot_scatter_plots(dataframe, target, path_to_save):

    print("-"*50)
    print("Target variable: **{}** .Analyzing variable for dataset with {} rows. ".format(target, len(dataframe)))
    print("-"*50)
    print("\n"*5)

    independent_variables = [col for col in list(
        dataframe.select_dtypes([int, float])) if col not in target]

    with PdfPages(path_to_save) as pdf:
        for variable in independent_variables:
            fig = plt.figure(figsize=(22, 5))
            title = fig.suptitle(
                "Analyzing variable: {}".format(variable), fontsize=14)
            fig.subplots_adjust(top=0.85, wspace=0.3)

            ax1 = fig.add_subplot(1, 5, 1)
            correlation = round(np.corrcoef(
                dataframe[variable], dataframe[target])[0, 1], 2)
            ax1.set_title('Correlation: {}'.format(correlation))
            sns.regplot(x=dataframe[variable], y=dataframe[target],
                        fit_reg=True, ax=ax1, scatter_kws={'alpha': 0.1})

            ax2 = fig.add_subplot(1, 5, 2)
            ax2.set_title("Distribution of variable")
            ax2.set_xlabel(variable)
            ax2.set_ylabel("Frequency")
            w_freq, w_bins, w_patches = ax2.hist(x=dataframe[variable], color='red', bins=15,
                                                 edgecolor='black', linewidth=1)

            ax3 = fig.add_subplot(1, 5, 3)
            ax3.set_title("Boxplot of variable")
            sns.boxplot(x=dataframe[variable], ax=ax3)

            data = dataframe[variable]
            data_without_outliers = data[abs(
                data - np.mean(data)) < 2 * np.std(data)]

            ax4 = fig.add_subplot(1, 5, 4)
            ax4.set_title("Distribution of variable dealing with outlers")
            ax4.set_xlabel(variable)
            ax4.set_ylabel("Frequency")
            w_freq, w_bins, w_patches = ax4.hist(x=data_without_outliers, color='red', bins=15,
                                                 edgecolor='black', linewidth=1)

            ax5 = fig.add_subplot(1, 5, 5)
            ax5.set_title("Boxplot of variable dealing with outliers")
            sns.boxplot(x=data_without_outliers, ax=ax5)

            pdf.savefig()
            plt.show()
            plt.close()


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_results_time_series(df, time_period_col, model, target, path_to_save_report, max_features=None, plot_since_period=0):

    mean_error = []

    with PdfPages(path_to_save_report) as pdf:
        for period in range(df[time_period_col].min()+1, df[time_period_col].max() + 1):

            train = df[df.time_period < period]
            test = df[df.time_period == period]

            X_train, X_test = train.drop(target, 1), test.drop(target, 1)
            y_train, y_test = train[target], test[target]

            #model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            error = rmse(y_test, y_pred)

            mean_error.append(error)

            if period >= plot_since_period:

                fig = plt.figure(figsize=(22, 5))
                title = fig.suptitle('Period {} - Error {} - Train size: {} / Test size: {}'.format(
                    period, round(error, 5), len(y_train), len(y_test)), fontsize=14)
                fig.subplots_adjust(top=0.85, wspace=0.1)

                ax1 = fig.add_subplot(1, 2, 1)

                visualizer = PredictionError(model, ax=ax1, line_color="red")
                visualizer.score(X_test, y_test)
                visualizer.finalize()

                ax2 = fig.add_subplot(1, 2, 2)
                visualizer = ResidualsPlot(model, ax=ax2)
                visualizer.fit(X_train, y_train)
                visualizer.score(X_test, y_test)
                visualizer.finalize()

                #ax3 = fig.add_subplot(1,3,3)
                #visualize.plot_coefficients(model, X_train)

                # plt.show()
                pdf.savefig(fig)
                plt.close()

                _logger.info('Period %d - Error %.5f' % (period, error))

            else:
                _logger.info('Period %d - Error %.5f' % (period, error))

    _logger.info('Mean Error = %.5f' % np.mean(mean_error))

    return model, X_train, y_train, X_test, y_test, mean_error


def plot_coefficients(model, X_train, max_features=None):
    """
        Plots sorted coefficient values of the model
    """

    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    if max_features:
        coefs = coefs[:max_features]

    plt.figure(figsize=(15, 7))
    plt.title("Coefficients importance")
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')

# .- Classification functions ---------------------------------------------------


def success_plot(x, y, title="", y_mean=None, pdf_path=None):
    try:
        x_names = list(np.unique(x))
    except:
        x_names = list(x.unique())
    x_ids = list(range(len(x_names)))

    x_base = np.array([x_ids[x_names.index(v)] for v in x])

    x_vals = x_ids
    x_counts_per_group = np.bincount(x_base)
    x_freq = x_counts_per_group/len(x_base)
    y_freq = [np.mean(y[x_base == i]) for i in x_vals]

    fig, ax1 = plt.subplots()

    color = 'tab:grey'
    ax1.set_xlabel('value')
    ax1.set_ylabel('volume', color=color)
    bar_plot = ax1.bar(x_vals, x_freq, color=color)

    for idx, rect in enumerate(bar_plot):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.,
                 height/2,  # vertical
                 x_counts_per_group[idx],
                 ha='center', va='bottom', rotation=0)

    if y_mean:
        ax1.axhline(y=y_mean, xmin=0, xmax=1)
        ax1.annotate('Total volume of positive class', xy=(0.5, y_mean), xytext=(0.5, y_mean + 0.2),
                     arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1, fc="0.6", ec="none"))
    #ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1])

    plt.xticks(x_vals, x_names, rotation=-25, ha='left')

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Churn ratio', color=color)
    ax2.plot(x_vals, y_freq, color=color, marker='o')
    ax2.set_ylim([0, 1])
    #ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if pdf_path:
        pdf_path.savefig()

    plt.show()


def success_plot_cont_var(x, y, title="", n_bins=10, y_mean=None, pdf_path=None):
    # Filter nans
    x_values = np.array(list(filter(lambda v: not np.isnan(v), x)))
    # calculate corr
    correlation = round(np.corrcoef(x_values, y[~x.isnull()])[0, 1], 2)
    # Calculate var bounds
    qs = np.percentile(x_values, [0, 25, 50, 75, 100])
    iqr = qs[3] - qs[1]
    x_min = max(min(x_values), qs[1]-1.5*iqr)
    x_max = min(max(x_values), qs[3]+1.5*iqr)
    # Calculate bins
    bin_size = (x_max - x_min)/n_bins
    bin_bounds = [x_min+i*bin_size for i in range(n_bins)]
    if (bin_bounds[0] > min(x_values)):
        bin_bounds = [min(x_values)] + bin_bounds
    if (bin_bounds[-1] < max(x_values)):
        bin_bounds = bin_bounds + [max(x_values)]

    # Calculate x segments
    x_segment = []
    for x_val in x:
        if np.isnan(x_val):
            x_segment.append("desc")
        else:
            s = None
            bin_pos = 1
            while (s == None and bin_pos < len(bin_bounds)):
                if (x_val >= bin_bounds[bin_pos-1]) and (x_val <= bin_bounds[bin_pos]):
                    s = "C{0} [{1}, {2}]".format(
                        str(bin_pos).zfill(2),
                        round(bin_bounds[bin_pos-1], 2),
                        round(bin_bounds[bin_pos], 2))
                    #s = "C{}".format(bin_pos)
                bin_pos += 1
            x_segment.append(s)

    x_segment = np.array(x_segment, dtype=np.str)
    title = "{} (correlation with target = {})".format(title, correlation)
    success_plot(x_segment, y, title, y_mean, pdf_path)


def report_univariate_churn(df, path, target, columns_to_plot, bins=10):
    mean_variable = (df[[target]].value_counts()/len(df)).loc[1].values[0]
    with PdfPages(path + '/Charts_binning.pdf') as export_pdf:
        for column in columns_to_plot:
            if df[column].dtype in [int, float]:
                try:
                    visualize.success_plot_cont_var(
                        df[column], df[target], title=column, y_mean=mean_variable, n_bins=bins, pdf_path=export_pdf)
                except:
                    print(column, "Error")
            else:
                try:
                    visualize.success_plot(
                        df[column], df[target], title=column, y_mean=mean_variable, pdf_path=export_pdf)
                except:
                    print(column, "Error")


def cumulative_gain_curve(y_true, y_score, pos_label=None):
    """This function generates the points necessary to plot the Cumulative Gain
    Note: This implementation is restricted to the binary classification task.
    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.
        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).
        pos_label (int or str, default=None): Label considered as positive and
            others are considered negative
    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.
        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.
    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains


def plot_lift_curve(y_true, y_probas, title='Lift Curve',
                    ax=None, figsize=None, title_fontsize="large",
                    text_fontsize="medium"):
    """Generates the Lift Curve from labels and scores/probabilities
    The lift curve is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html.
    The implementation here works only for binary classification.
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        title (string, optional): Title of the generated plot. Defaults to
            "Lift Curve".
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.metrics.plot_lift_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_lift_curve.png
           :align: center
           :alt: Lift Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Lift Curve for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, y_probas[:, 0],
                                                classes[0])
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1],
                                                classes[1])

    percentages = percentages[1:]
    gains1 = gains1[1:]
    gains2 = gains2[1:]

    gains1 = gains1 / percentages
    gains2 = gains2 / percentages

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(percentages, gains1, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(percentages, gains2, lw=3, label='Class {}'.format(classes[1]))

    ax.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Lift', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid(True)
    ax.legend(loc='lower right', fontsize=text_fontsize)

    return ax


def classification_model_report_summary(data: pd.core.frame.DataFrame,
                                        target: pd.core.frame.DataFrame,
                                        model: sklearn.pipeline.Pipeline,
                                        title_report: str = "Model report",
                                        classes: List[str] = None,
                                        classes_dict: Dict[int, str] = None) -> matplotlib.figure.Figure:
    """Generate a report for visualizing some important metrics performance about a specific classification model

    Args:
        data (pd.core.frame.DataFrame): pandas dataframe with the input variables
        target (pd.core.frame.DataFrame): pandas series with the target
        model (sklearn.pipeline.Pipeline): classification model pretrained
        title_report (str, optional): The main title for the figure. Defaults to "Model report".
        classes (List[str], optional):  The class labels to use for the legend ordered by the index 
        of the sorted classes discovered in the fit() method. 
        Specifying classes in this manner is used to change the class names to a more specific 
        format or to label encoded integer classes. Some visualizers may also use this field to 
        filter the visualization for specific classes. For more advanced usage specify an encoder 
        rather than class labels.. Defaults to None.
        classes_dict (Dict[int, str], optional): The class labels to use for the legend ordered by 
        the index of the sorted classes discovered in the fit() method. Specifying classes in 
        this manner is used to change the class names to a more specific format or to label encoded integer classes. 
        Some visualizers may also use this field to filter the visualization for specific classes. 
        For more advanced usage specify an encoder rather than class labels.. Defaults to None.

    Returns:
        matplotlib.figure.Figure
    """                                        

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        fig = plt.figure(figsize=(20, 10))
        grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.3)

        ax1 = plt.subplot(grid[0, 0:2])
        yb.classifier.classification_report(model=model,
                                            X_train=data,
                                            y_train=target,
                                            classes=classes,
                                            is_fitted=True,
                                            show=False,
                                            support="count",
                                            encoder=classes_dict,
                                            ax=ax1)
        ax1.set_title("Model Report Summary", fontsize='xx-large')

        ax2 = plt.subplot(grid[0, 2])
        yb.classifier.confusion_matrix(model=model,
                                       X_train=data,
                                       y_train=target,
                                       classes=classes,
                                       percent=False,
                                       is_fitted=True,
                                       show=False,
                                       ax=ax2)
        ax2.set_title("Confussion Matrix", fontsize='xx-large')

        ax3 = plt.subplot(grid[0, 3])
        yb.classifier.confusion_matrix(model=model,
                                       X_train=data,
                                       y_train=target,
                                       classes=classes,
                                       percent=True,
                                       is_fitted=True,
                                       show=False,
                                       title="Confussion Matrix normalized",
                                       ax=ax3)
        ax3.set_title("Confussion Matrix Normalized", fontsize='xx-large')

        ax4 = plt.subplot(grid[1, 0:2])
        yb.classifier.roc_auc(model=model,
                              X_train=data,
                              y_train=target,
                              classes=classes,
                              is_fitted=True,
                              show=False,
                              ax=ax4
                              )
        ax4.set_title("Roc AUC curve", fontsize='xx-large')
        ax4.set_ylim([-0.02, 1.02])
        ax4.set_xlim([-0.02, 1.02])

        ax5 = plt.subplot(grid[1, 2:])
        visualize.plot_lift_curve(target, model.predict_proba(data), title='Lift Curve',
                                  ax=ax5, figsize=None, title_fontsize="large",
                                  text_fontsize="medium")
        ax5.set_title("Lift Curve", fontsize='xx-large')
        fig.suptitle(title_report, fontsize='xx-large')

#.- Cluster Functions ########################################


def get_summary_cluster(df, cluster_labels, cluster_id, categorical_columns_names):
    cluster_df = df[cluster_labels == cluster_id]
    summary_cluster = (cluster_df.mean().to_dict())

    if len(categorical_columns_names) > 0:
        summary_cluster.update(
            cluster[categorical_columns_names].mode().to_dict(orient="records")[0])

    summary_cluster["cluster_id"] = cluster_id
    summary_cluster["n_values"] = len(cluster_df)

    return summary_cluster


def comparare_clusters(df, cluster_labels, categorical_columns_names, cluster_ids):
    summaries = []
    for cluster_id in cluster_ids:
        summaries.append(get_summary_cluster(df=df,
                                             cluster_labels=cluster_labels,
                                             categorical_columns_names=categorical_columns_names,
                                             cluster_id=cluster_id))

    return pd.DataFrame(summaries).set_index("cluster_id").T
