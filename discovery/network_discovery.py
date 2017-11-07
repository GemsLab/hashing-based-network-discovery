"""
Network discovery methods and discovered network analysis
"""
import random
import inspect
import math
import itertools
import numpy as np
import networkx as nx
from sklearn import preprocessing
from scipy import spatial, stats
from collections import defaultdict

def binarize(data):
    """Binarizes s.t. x[i] = 1 if x[i] > mean(x) else 0"""
    data = np.array(data)
    binary_data = np.zeros(data.shape, dtype=np.int8)
    dimensions = len(data.shape)
    mean = np.mean(data, axis=0 if dimensions == 1 else 1)

    if dimensions == 1:
        for i in range(len(data)):
            if data[i] > mean:
                binary_data[i] = 1
    else:
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                if data[i, j] > mean[i]:
                    binary_data[i, j] = 1
    return binary_data

def normalize(data):
    """Subtract mean from data"""
    return data - np.mean(data)

def z_normalize(data):
    """Z normalizes data"""
    dimensions = len(data.shape)
    if dimensions == 1:
        mu, sigma = np.mean(data), np.std(data)
    else:
        mu, sigma = np.mean(data, axis=0), np.std(data, axis=0)
    return (data - mu) / sigma

def abc_similarity(x, y, alpha=.0001):
    """Assume binarized data"""
    total = exp = same = 0
    base, n = 1 + alpha, len(x)
    S = (base ** n - 1) / alpha
    for i in range(n):
        if x[i] == y[i]:
            total += base ** exp
            same += 1
            exp += 1
        else:
            exp = 0
    return 1 if same == n else total / S

def abc_distance(x, y, alpha=.0001):
    """Assume binarized data"""
    total = exp = same = 0
    base, n = 1 + alpha, len(x)
    S = ((1 + alpha) ** n - 1) / alpha
    for i in range(n):
        if x[i] == y[i]:
            total += base ** exp
            same += 1
            exp += 1
        else:
            exp = 0
    return 0 if same == n else S - total

def pearson_correlation(x, y):
    """Absolute-valued pearson correlation coefficient"""
    return abs(stats.pearsonr(x, y)[0])

def get_pairwise_weights(sim_method, data, pairs, theta=0):
    """Gets all pairwise weights in the pairs iterable and returns
    a networkx graph object"""
    all_weights = {}
    pairs = list(pairs)
    print('Number of comparisons: {0}'.format(len(pairs)))
    for x, y in pairs:
        all_weights[(x, y)] = sim_method(data[x, :], data[y, :])

    G = nx.Graph()
    for pair, weight in all_weights.items():
        if weight > theta:
            G.add_edge(*pair, weight=weight)
    return G

def pairwise(sim_method, data, theta=0):
    """
    Returns a dictionary of (node i, node j): weight pairs
    created by all-pairs comparisons of rows of data.

    Parameters
    ----------
    theta : float in [0, 1]
       Threshold value above which indices are returned for.
    """
    all_pairs = itertools.combinations(range(data.shape[0]), 2)
    return get_pairwise_weights(sim_method, data, all_pairs, theta=theta)

def bitlist_to_int(l):
    """Converts a list of bits to an int"""
    num = 0
    l = list(map(int, l))
    for bit in l:
        num = (num << 1) | bit
    return num

def window_lsh(sim_method, data, k=3, r=2, b=4):
    """
    Performs hashing of the input data by window sampling.

    Parameters
    ----------
    data : np.ndarray
        Input data
    r : integer
        Number of windows per hash signature (AND construction)
    b : integer
        Number of hash tables to construct (OR construction)
    """
    n = data.shape[1]
    indices = [random.sample(range(n - k + 1), r) for _ in range(b)]
    tables = [defaultdict(list) for _ in range(b)]
    all_pairs = set()

    for t in range(len(tables)):
        table = tables[t]
        current = indices[t]

        for x in range(len(data)):
            row = data[x, :]
            signature = tuple([bitlist_to_int(row[index:index + k]) for index in current])
            for node in table[signature]:
                pair = (min(x, node), max(x, node))
                all_pairs.add(pair)

            table[signature].append(x)

    return get_pairwise_weights(sim_method, data, all_pairs)

def plot_adj_mat(A, show=True, save_file=None):
    """Simple adjacency matrix plot"""
    plt.matshow(G.A, fignum=100, cmap='RdPu')

    if show:
        plt.show()
    if save_file is not None:
        plt.draw()
        plt.savefig(save_file)

def avg_degree(G):
    """Average weighted degree"""
    if not G.number_of_nodes():
        return 0
    degree_sum = sum([pair[1] for pair in G.degree])
    return degree_sum / G.number_of_nodes()

def avg_weighted_degree(G):
    """Average weighted degree"""
    if not G.number_of_nodes():
        return 0
    degree_sum = sum([pair[1] for pair in G.degree(weight='weight')])
    return degree_sum / G.number_of_nodes()

def avg_cc(G, ratio=.2):
    """Optionally samples a ratio of the nodes"""
    if ratio is None:
        return nx.average_clustering(G, weight='weight')

    nodes = G.nodes()
    nodes = random.sample(nodes, math.ceil(ratio * len(nodes)))
    return nx.average_clustering(G, nodes=nodes, weight='weight')

def modularity(G):
    """Uses Louvain community detection to calculate modularity"""
    partition = community.best_partition(G)
    return community.modularity(partition, G)

def avg_path_length(G):
    """Average unweighted path length"""
    avgs = [ nx.average_shortest_path_length(g) for g in nx.connected_component_subgraphs(G) ]
    avg = sum(avgs) / len(avgs)
    return avg



