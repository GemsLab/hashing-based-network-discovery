"""
For file I/O on:
    - numpy matrices
    - pandas DataFrames
    - mat files
    - networkx graphs
    - etc
"""
import csv
import os
import json
import numpy as np
import scipy.io as sio
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

def get_lines_in_file(filename):
    """Return array of lines from file"""
    with open(filename) as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
        return lines

def data_as_np(filename, delimiter=',', skip_header=0):
    """Returns data as Numpy array"""
    return np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header)

def data_as_pd(filename, keep_columns=None, drop_columns=None):
    """Return data as a pandas DataFrame"""
    df = pd.read_csv(filename)
    if keep_columns is not None:
        df = df[keep_columns]
    if drop_columns is not None:
        df = df.drop(drop_columns, axis=1)

    return df

def nx_from_edgelist(filename, delimiter=',', ext='.csv'):
    return nx.read_weighted_edgelist(filename + ext, delimiter=delimiter, nodetype=int)

def parse_edgelist(filename, ext=''):
    """Parses line-by-line edgelist"""
    with open(filename + ext) as f:
        lines = get_lines_in_file(filename)
        return nx.parse_edgelist(lines, nodetype=int)

def data_from_mat(filename):
    """Get data from .mat file"""
    return _load_mat(filename)

def _load_mat(filename):
    """
    This function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.

    Source for this function and the functions it calls:
    http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(d):
    """
    Checks if entries in dictionary are mat-objects. If yes,
    todict is called to change them to nested dictionaries.
    """
    for key in d:
        if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
    return d

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries.
    """
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        else:
            d[strg] = elem
    return d

def write_matrix_to_csv(filename, data):
    """Writes whole matrix to CSV"""
    np.savetxt(filename, data, delimiter=',')

def write_row_to_csv(filename, row):
    """row is a list of items to write to the csv"""
    with open(filename, 'a+') as f:
        f.write(_list_to_csv_string(row))

def write_rows_to_csv(filename, rows, header=None):
    """Writes a list of lists (rows) to the CSV with an optional header"""
    with open(filename, 'w') as f:
        if header is not None:
            f.write(_list_to_csv_string(header))
        for row in rows:
            f.write(_list_to_csv_string(row))

def write_column_to_csv(filename, data, header):
    """Adds a column to a CSV file"""
    csv_input = pd.read_csv(filename)
    csv_input[header] = data
    csv_input.to_csv(filename, index=False)

def _list_to_csv_string(l, delimiter=','):
    """List to delimited string with newline"""
    return delimiter.join(map(str, l)) + '\n'

def write_edgelist(filename, G, delimiter=',', ext='.csv'):
    """Writes the edge list to a CSV file"""
    nx.write_weighted_edgelist(G, filename + ext, delimiter=delimiter)

def write_json_edgelist(filename, G, ext='.json'):
    """Converts nx graph to JSON and writes"""
    json_data = json_graph.node_link_data(G)
    s = json.dumps(json_data)
    with open(filename + ext, 'w') as f:
        f.write(s)
