# -*- coding: utf-8 -*-
"""
Utility Functions

@author: eking
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib



def load_data(selector="motions"):
    
    if selector == 'motions':

        # Load Data
        raw_motions = pd.read_csv("datasets//motions.csv", header = None, delimiter=",")
        raw_motions.head()
        
        # Rename response into something sensical
        class_names=['rock', 'paper', 'scissors', 'okay']
        for i in range(4):
            raw_motions[64].replace(i, class_names[i], inplace=True)
        
        raw_motions.columns = [str(i) for i in raw_motions.columns]
        raw_motions.columns = raw_motions.columns.str.replace('64', 'motion_type')
        #raw_motions.head()
        
        # Split into response and predictors
        y = raw_motions[['motion_type']]
        X = raw_motions.drop(['motion_type'], axis = 1)
        print(X.shape, y.shape)
        
        # Randomly divide into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
    if selector == 'particles':

        # Load Data
        df = pd.read_csv("datasets//smaller_particles.csv", header = 0, delimiter=",")
        df.head()
        
        # Set class names
        class_names=['proton', 'pion', 'kaon', 'positron']
        
        # Split into response and predictors
        y = df[['id']]
        X = df.drop(['id'], axis = 1)
        #print(X.shape, y.shape)
        
        # Randomly divide into train and test sets
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 42)
            
    
    return X_train, X_test, y_train, y_test, class_names
    
def plot_learning_curves(estimator, X_train, y_train, title = "Learning Curve", cv = 10, scorer = 'accuracy', low_limit = 0.8):
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train.values.ravel('C'), cv=cv, scoring = scorer)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = "Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title = (title)
    plt.ylim(low_limit, 1)
    plt.legend(loc='best')
    plt.show()
    return 1


# Utility function to report best scores
def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
            
def plot_bars(scores_list_m, time_list_m, scores_list_p, time_list_p, test_parameter, n_range):
    n_groups = len(n_range)
    index = np.arange(n_groups)
    bar_width=0.35
    opacity=1
    
    plt.bar(index, scores_list_m, bar_width, 
                     alpha = opacity,
                     color = 'blue',
                     label = 'Motions')
    plt.bar(index + bar_width, scores_list_p, bar_width,
                     alpha = opacity,
                     color = 'green',
                     label = 'Particles')

    plt.xlabel(test_parameter)
    plt.ylabel('Validation set accuracy')
    plt.xticks(index + bar_width, n_range)
    plt.legend()
    plt.show()
    
    plt.bar(index, time_list_m, bar_width, 
                     alpha = opacity,
                     color = 'blue',
                     label = 'Motions')
    plt.bar(index + bar_width, time_list_p, bar_width,
                     alpha = opacity,
                     color = 'green',
                     label = 'Particles')
    
    plt.xlabel(test_parameter)
    plt.ylabel('Computation Time')
    plt.xticks(index + bar_width, n_range)
    plt.legend()
    plt.show()
    

def plot_lines(scores_list_m, time_list_m, scores_list_p, time_list_p, test_parameter, n_range):
    plt.plot(n_range, scores_list_m, color='blue', label='Motions')
    plt.plot(n_range, scores_list_p, color='green', label='Particles')
    plt.xlabel(test_parameter)
    plt.ylabel('Validation set accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(n_range, time_list_m, color='blue', label='Motions')
    plt.plot(n_range, time_list_p, color='green', label='Particles')
    plt.xlabel(test_parameter)
    plt.ylabel('Computation Time')
    plt.legend()
    plt.show()
    

def plot_lines1(scores_list, time_list, test_parameter, n_range, col = 'blue', label = 'Motions'):
    plt.plot(n_range, scores_list, color=col, label=label)
    plt.xlabel(test_parameter)
    plt.ylabel('Validation set accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(n_range, time_list, color=col, label=label)
    plt.xlabel(test_parameter)
    plt.ylabel('Computation Time')
    plt.legend()
    plt.show()
    

#From https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", xlabel='x-var', ylabel='y-var', **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar



def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts