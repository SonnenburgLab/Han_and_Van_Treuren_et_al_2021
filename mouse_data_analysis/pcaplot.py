import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAPlot:

    """
    Conduct Principal Component Analysis (PCA) to separate metabolomic profiles by sample types
    or colonization states.

    """

    COLONIZATION_COLOR_MAP = {
        'Bt_Ca_Er_Pd_Et': 'tab:purple',
        'Cs_Bt_Ca_Er_Pd_Et': 'black',
        'Bt': 'tab:blue',
        'Cs': 'tab:orange',
        'germ-free': 'tab:gray',
    }

    SAMPLE_TYPE_COLOR_MAP = {
        'caecal': 'tab:brown',
        'feces': 'black',
        'serum': 'tab:red',
        'urine': 'tab:orange',
    }

    def __init__(self, fc_matrix, metadata, colonizations, sample_types):
        if len(colonizations) > 1 and len(sample_types) > 1:
            raise Exception('Can only color PCA plot by either colonizations or sample types')

        self.colonizations = colonizations
        self.sample_types = sample_types

        self.fc_matrix = fc_matrix \
            .join(metadata[['experiment', 'sample_type', 'colonization']])

        self.fc_matrix = self.fc_matrix[
            (self.fc_matrix['sample_type'].isin(sample_types)) &
            (self.fc_matrix['colonization'].isin(colonizations))
        ]

        feature_matrix = self.fc_matrix \
            .drop(columns=['experiment', 'colonization', 'sample_type']) \
            .dropna(axis='columns', how='any')

        x = feature_matrix.values

        # print('Feature matrix:')
        # print(x)

        x = StandardScaler().fit_transform(x)

        # print('After scaling to mean = 0 and variance = 1:')
        # print(x)

        pca = PCA(n_components=3)

        principalComponents = pca.fit_transform(x)

        # print('Component variances:')
        # print(pca.explained_variance_ratio_)

        principalDf = pd.DataFrame(data=principalComponents,
                                   columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3'])

        principalDf['colonization'] = self.fc_matrix['colonization'].values
        principalDf['sample_type'] = self.fc_matrix['sample_type'].values

        # print(principalDf)

        self.pca = pca
        self.principalDf = principalDf


    def plot_components(self, components):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        ax.tick_params(
            axis='both',
            labelsize = 20
        )

        ax.set_xlabel(
            'Principal Component {} ({:.1f}%)'.format(components[0], self.pca.explained_variance_ratio_[components[0] - 1] * 100),
            fontsize = 20
        )

        ax.set_ylabel(
            'Principal Component {} ({:.1f}%)'.format(components[1], self.pca.explained_variance_ratio_[components[1] - 1] * 100),
            fontsize = 20
        )

        if len(self.colonizations) > 1:
            colors = self.fc_matrix['colonization'].map(self.COLONIZATION_COLOR_MAP)

            custom_lines = list(map(
                lambda colonization: Line2D([0], [0], color=self.COLONIZATION_COLOR_MAP[colonization], lw=4),
                self.colonizations
            ))
        else:
            colors = self.fc_matrix['sample_type'].map(self.SAMPLE_TYPE_COLOR_MAP)

            custom_lines = list(map(
                lambda sample_type: Line2D([0], [0], color=self.SAMPLE_TYPE_COLOR_MAP[sample_type], lw=4),
                self.sample_types
            ))

        ax.scatter(
            self.principalDf['Principal Component {}'.format(components[0])],
            self.principalDf['Principal Component {}'.format(components[1])],
            c=colors
        )

        ax.legend(custom_lines, self.colonizations if len(self.colonizations) > 1 else self.sample_types)

        return (fig, ax)


    def plot3d(self):
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel(
            'Principal Component 1 ({:.1f}%)'.format(self.pca.explained_variance_ratio_[0] * 100),
            fontsize = 12
        )

        ax.set_ylabel(
            'Principal Component 2 ({:.1f}%)'.format(self.pca.explained_variance_ratio_[1] * 100),
            fontsize = 12
        )

        ax.set_zlabel(
            'Principal Component 3 ({:.1f}%)'.format(self.pca.explained_variance_ratio_[2] * 100),
            fontsize = 12
        )

        if len(self.colonizations) > 1:
            colors = self.fc_matrix['colonization'].map(self.COLONIZATION_COLOR_MAP)

            custom_lines = list(map(
                lambda colonization: Line2D([0], [0], color=self.COLONIZATION_COLOR_MAP[colonization], lw=4),
                self.colonizations
            ))
        else:
            colors = self.fc_matrix['sample_type'].map(self.SAMPLE_TYPE_COLOR_MAP)

            custom_lines = list(map(
                lambda sample_type: Line2D([0], [0], color=self.SAMPLE_TYPE_COLOR_MAP[sample_type], lw=4),
                self.sample_types
            ))

        ax.scatter(
            self.principalDf['Principal Component 1'],
            self.principalDf['Principal Component 2'],
            self.principalDf['Principal Component 3'],
            c=colors
        )

        ax.legend(custom_lines, self.colonizations if len(self.colonizations) > 1 else self.sample_types)

        return (fig, ax)
