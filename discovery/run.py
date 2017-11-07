"""
Use the graph generation methods provided by the network_discovery module
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from time import time
from scipy.sparse import csr_matrix
from network_discovery import *
from dataset import *
from file_io import *

def classify(model, X, y, cv):
    """
    Parameters
    ----------
    model : sklearn classification object
    X : numpy array (n_samples, n_classes)
        The samples
    y : numpy array (n_samples,)
        The labels
    cv : sklearn cross-validator
        The object that provides train/test split indices
    """
    perf_metrics = [ metrics.accuracy_score, metrics.precision_score, metrics.recall_score,
                     metrics.f1_score, metrics.roc_auc_score ]
    avg_scores = np.zeros(len(perf_metrics))
    n_splits = cv.get_n_splits(X)

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores = np.asarray([perf_metric(y_test, y_pred) for perf_metric in perf_metrics])
        avg_scores += scores

    avg_scores /= n_splits
    return list(avg_scores)

def create_feature_vector(graph):
    """Create a feature vector for the graph using well-known graph statistics"""
    awd = avg_weighted_degree(graph)
    cc = avg_cc(graph)
    mod = modularity(graph)
    apl = avg_path_length(graph)
    dens = nx.density(graph)
    return [ awd, cc, mod, apl, dens ]

def get_prediction_scores(graphs, labels):
    """
    Predict health status for different people.
    For each graph, create a feature vector out of that graph.

    Parameters
    ----------
    graphs : length-n list of networkx graphs
    labels : length-n list of binary (health) labels associated with the graphs
    """
    assert len(graphs) == len(labels)

    X = []
    for graph in graphs: # create a vector from each graph
        create_feature_vector(graph)
    X = np.asarray(X)
    y = labels

    C = 1 # logistic regression regularization parameter
    model = linear_model.LogisticRegression(C=C)
    cv = model_selection.StratifiedKFold(n_splits=10)
    scores = classify(model, X, y, cv)
    print('Average classification scores:')
    print('Accuracy: {}'.format(scores[0]))
    print('Precision: {}'.format(scores[1]))
    print('Recall: {}'.format(scores[2]))
    print('F1: {}'.format(scores[3]))
    print('AUC: {}'.format(scores[4]))

def main():
    # Here, use a randomly generated dataset
    # You can replace this with real time series data
    # Use the functions in the file_io and dataset modules to read in your data
    dataset = SyntheticDataset()
    data = dataset.generate_data(100, 1000, write_to_file=False)

    # Construct a graph using pairwise correlation, thresholding at .3
    # You can change the theta parameter to any number between 0 and 1
    theta = .3
    t0 = time()
    print('Constructing corr graph...')
    corr_graph = pairwise(pearson_correlation, data, theta=theta)
    print('Constructed graph in {:.2f} seconds'.format(time() - t0))
    print('Number of edges: {}'.format(
        corr_graph.number_of_edges()))
    print('Corr graph average weighted degree: {:.2f}'.format(
        avg_weighted_degree(corr_graph)))

    print()
    # Construct a graph using LSH, with a window size of 8
    # You can change the window size k, length of hash signatures r, and the
    # number of hash tables b
    k, r, b = 8, 1, 10
    data = binarize(data)
    t0 = time()
    print('Constructing LSH graph...')
    lsh_graph = window_lsh(abc_similarity, data, k=k, r=r, b=b)
    print('Constructed graph in {:.2f} seconds'.format(time() - t0))
    print('Number of edges: {}'.format(
        lsh_graph.number_of_edges()))
    print('LSH graph average weighted degree: {:.2f}'.format(
        avg_weighted_degree(lsh_graph)))

    # Generate n graphs with associated health labels, and use
    # labels in a classification task
    """
    cobre_data = COBREDataset(subdir='data/') # change to the subdirectory where your COBRE data live
    labels = cobre_data.labels
    graphs = []
    theta = .6
    for subject_id, data in cobre_data.gen_data():
        print('Generating graph for subject {}'.format(subject_id))
        graphs.append(pairwise(pearson_correlation, data, theta=theta))
    get_prediction_scores(graphs, labels)
    """

if __name__ == '__main__':
    main()
