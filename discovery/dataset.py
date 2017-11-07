"""
For keeping track of the different kinds of data in the project:
    - Raw fMRI data
    - Generated graph data
    - Evaluation data (statistics, feature vectors)
"""
import os
import numpy as np
import networkx as nx
from file_io import *

class Dataset(object):

    def __init__(self, subdir, data_file=None):
        """
        Parameters
        ----------
        subdir : str
            Where the data live (one file, multiple files, files of various extensions, etc)
        data_file : str
            Optionally, if all data live in a single file in the subdir
        """
        self.subdir = subdir
        self.data_file = data_file

    @property
    def filenames_with_paths(self):
        """For getting filenames with relative paths"""
        if self.subdir[-1] != '/':
            self.subdir += '/'
        return [self.subdir + f for f in sorted(os.listdir(self.subdir))]

    def subject_id_from_filename(f):
        """Subject id is before the file extension, separated by _"""
        return f.split('_')[-1].split('.')[0]

class UCRDataset(Dataset):

    def __init__(self, subdir, data_file):
        super().__init__(subdir, data_file=data_file)

    def read_data(self):
        labels, series = [], []
        if self.subdir[-1] != '/':
            self.subdir += '/'

        lines = get_lines_in_file(self.subdir + self.data_file)
        lines = [list(map(float, line.split(','))) for line in lines]
        labels = [line[0] for line in lines]
        data = [line[1:] for line in lines]

        return np.array(labels), np.array(data)

class FMRIDataset(Dataset):

    def __init__(self, subdir, data_file=None):
        super().__init__(subdir, data_file=data_file)

    def gen_data(self):
        """Returns a generator of subject_id, dataset pairs"""
        raise NotImplementedError

class COBREDataset(FMRIDataset):

    def __init__(self, subdir='data/cobre/',
                 data_file='Schiz_COBRE_1166_p50f0b_Danai.mat',
                 label_file='Schiz_COBRE_MDF_Danai.csv'):
        """All data are in a single mat file. Also have an associated label/annotation file"""
        super().__init__(subdir, data_file)
        self.label_file = label_file

    @property
    def labels(self):
        """Returns [<healthy ids>], [<unhealthy_ids>]"""
        subject, status = 'Subject', 'Dx'
        control, patient = 'Control', 'Patient'

        data = data_as_pd(self.subdir + self.label_file, [subject, status])
        id_column, health_column = data[subject], data[status]

        healthy = ['00' + str(id_column[i]) for i in range(len(data)) if health_column[i] == control]
        unhealthy = ['00' + str(id_column[i]) for i in range(len(data)) if health_column[i] == patient]
        return healthy, unhealthy

    def gen_data(self):
        data = data_from_mat(self.subdir + self.data_file)['data']
        for i in range(data.shape[0]):
            yield '{0}'.format(data[i].Subject), data[i].roiTC.T

class PennDataset(FMRIDataset):

    SCORE_COLUMNS = ['Complex Cognition', 'Memory', 'Social Cognition']

    def __init__(self, subdir='data/penn/',
                 data_file=None, score_file='penn_scores.csv'):
        super().__init__(subdir, data_file)
        self.score_file = score_file

    @property
    def scores(self):
        """Returns dictionary of { subject_id : [complex cognition, memory, social cognition] } scores"""
        data = data_as_pd(self.score_file)
        col1, col2, col3 = data[PennDataset.SCORE_COLUMNS[0]], \
            data[PennDataset.SCORE_COLUMNS[1]], \
            data[PennDataset.SCORE_COLUMNS[2]]
        subject_id_column = data['subject_id']

        subject_scores = {}
        for i in range(len(subject_id_column)):
            subject_id = str(int(subject_id_column[i]))
            subject_scores[subject_id] = [ col1[i], col2[i], col3[i] ]
        return subject_scores

    def gen_data(self):
        files = [f for f in self.filenames_with_paths if f[-3:] == 'mat']
        for mat in files:
            yield Dataset.subject_id_from_filename(mat), data_from_mat(mat)['roiTC'].T

class SyntheticDataset(Dataset):

    def __init__(self, subdir='data/', data_file='synth.txt'):
        super().__init__(subdir, data_file)

    def generate_data(self, N, num_samples, write_to_file=True):
        """Create some synthetic data, each sample of size N"""
        A = np.random.rand(N, N)
        cov = np.dot(A, A.T)
        mean = np.zeros(N)
        data = np.random.multivariate_normal(mean, cov, size=num_samples)
        if write_to_file:
            write_matrix_to_csv(self.subdir + self.data_file, data)
        return data

class GraphDataset(Dataset):

    def __init__(self, subdir):
        super().__init__(subdir, data_file=None)

    def gen_graphs(self, ext='', dict_format=False):
        """Returns a generator of subject_id, nx graph pairs"""
        files = self.filenames_with_paths
        for f in files:
            subject_id = Dataset.subject_id_from_filename(f)
            G = parse_edgelist(f, ext=ext) if dict_format else nx_from_edgelist(f, ext=ext)
            yield subject_id, G

class COBREGraphDataset(GraphDataset):
    """Generated COBRE graphs"""

    GRAPH_DIR = '/y/DATA/schiz_graphs/'

    def __init__(self, subdir):
        super().__init__(COBREGraphDataset.GRAPH_DIR + subdir)

class PennGraphDataset(GraphDataset):
    """Generated Penn graphs"""

    GRAPH_DIR = '/y/DATA/penn_graphs/'

    def __init__(self, subdir):
        super().__init__(PennGraphDataset.GRAPH_DIR + subdir)

class FeatureDataset(Dataset):

    def __init__(self, subdir, data_file):
        super().__init__(subdir, data_file)

    def _split_ids_and_features(self, id_column):
        """Returns the ID column as a separate DF from the features"""
        df = data_as_pd(self.subdir + self.data_file)
        subject_id_column = df[id_column]
        X = df.drop([id_column], axis=1).as_matrix() # feature vectors
        return subject_id_column, X

    @property
    def X_y(self):
        """Return feature vectors and label(s)"""
        raise NotImplementedError

class COBREFeatureDataset(FeatureDataset):

    def __init__(self, subdir, data_file):
        super().__init__(subdir, data_file)

    @property
    def X_y(self):
        """Returns X (features), y (target).
        Features are BINARY"""
        subject_id_column, X = self._split_ids_and_features('subject_id')

        y = np.zeros(len(subject_id_column))
        healthy, unhealthy = COBREDataset().labels
        healthy, unhealthy = set(healthy), set(unhealthy)

        for i in range(len(subject_id_column)):
            subject_id = '00{0}'.format(subject_id_column[i])
            if subject_id in healthy:
                y[i] = 1
            elif subject_id in unhealthy:
                y[i] = 0
        return X, y

def main():
    pass

if __name__ == '__main__':
    main()
