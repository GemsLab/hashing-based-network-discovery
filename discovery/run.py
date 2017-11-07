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

def main():
    dataset = SyntheticDataset()
    data = dataset.generate_data(100, 1000, write_to_file=False)

    t0 = time()
    print('Constructing corr graph...')
    corr_graph = pairwise(pearson_correlation, data, theta=.3)
    print('Constructed graph in {:.2f} seconds'.format(time() - t0))
    print('Number of edges: {}'.format(
        corr_graph.number_of_edges()))
    print('Corr graph average weighted degree: {:.2f}'.format(
        avg_weighted_degree(corr_graph)))

    print()
    data = binarize(data)
    t0 = time()
    print('Constructing LSH graph...')
    lsh_graph = window_lsh(abc_similarity, data, k=8, r=1, b=10)
    print('Constructed graph in {:.2f} seconds'.format(time() - t0))
    print('Number of edges: {}'.format(
        lsh_graph.number_of_edges()))
    print('LSH graph average weighted degree: {:.2f}'.format(
        avg_weighted_degree(lsh_graph)))

if __name__ == '__main__':
    main()
