# Normal utilities
import numpy as np
import pandas as pd

# For building trees
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 


def tree_state_as_df(clf, tree):
    '''Adapted from the extremely helpful:
    https://stackoverflow.com/questions/32506951/how-to-explore-a-decision-tree-built-using-scikit-learn
    
    clf = trained classifier
    ex
    rf = RandomForestClassifier(n_estimators=50, max_depth=7, bootstrap=False,
                                max_features=100)
    rf.fit(train_data, train_labels)
    preds = rf.predict(test_data)
    tree_state_as_df(rf, rf.estimators[5]) # using 6th tree in the forest
    '''
    # tree.tree_.__getstate__()['nodes'] is a named tuple array
    tmp = pd.DataFrame(tree.tree_.__getstate__()['nodes'])
    tmp['node_type'] = np.where((tmp['left_child'] != -1).values, 'internal',
                                'leaf')
    # Would call them test nodes, but hard to read as leaf, test by eye.

    # For leaf nodes there is no feature or threshold. Remove these.
    idxs = (tmp['feature'] == -2)
    tmp.loc[idxs, 'feature'] = np.nan
    tmp.loc[idxs, 'threshold'] = np.nan

    # Take the classes the tree was trained on and record the distribution at
    # each node as well as the dominant (e.g. what that node counts as).
    for cl, v in zip(clf.classes_, tree.tree_.value.squeeze().T):
        tmp[cl] = v
    node_dom_class = clf.classes_[np.argmax(tmp[clf.classes_].values, 1)]
    tmp['node_class'] = node_dom_class

    # Name the index.
    tmp.index.name = 'node'
    return tmp

def test_sample_fate_by_leaf(tree, test_data, test_labels):
    '''Determine the final nodes (leaves) of the test samples.

    Note: the leaf id's (the row indices of this dataframe, correspond to the
    index in `tree.tree_.__getstate__()['nodes']`.
    '''
    # Note, this doesn't represent all the leaves in the tree, just the ones
    # these particular test samples go to.
    leaf_ids = tree.apply(test_data)
    leaves = sorted(set(leaf_ids))
    leaf_to_idx = {leaf:i for i, leaf in enumerate(leaves)}

    # Similarly, these are just the classes seen in the test_labels
    classes = sorted(set(test_labels))
    class_to_idx = {_class:i for i, _class in enumerate(classes)}

    results = np.zeros((len(leaves), len(classes)))

    for label, leaf in zip(test_labels, leaf_ids):
        results[leaf_to_idx[leaf], class_to_idx[label]] += 1

    return pd.DataFrame(results, index=leaves, columns=classes).T

def follow_decisions(tree, test_data):
    '''
    adapted from
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    '''

    # The path each sample takes
    node_paths = tree.decision_path(test_data)

    # The leaf that each sample ends up in
    predicted_leaves = tree.apply(test_data)

    # The feature used to decide at each node
    features = tree.tree_.feature

    # The threshold for the deciding feature at each node. If the value of the
    # sample is smaller than the threshold we move to left child, else right.
    thresholds = tree.tree_.threshold

    # For dataframe creeation balow
    columns = ['feature', 'threshold', 'sample_value']

    results = {}
    for idx in range(len(test_data.index)):

        # Don't understand the syntax, but this is clearly the path in nodes,
        # terminating in a leaf node, that this particular sample takes.
        node_indices = node_paths.indices[\
            node_paths.indptr[idx]: node_paths.indptr[idx + 1]]

        tmp_results = np.zeros((len(node_indices), 4))
        tmp_results[:, 0] = node_indices
        tmp_results[:, 1] = features[node_indices]
        tmp_results[:, 2] = thresholds[node_indices]
        tmp_results[:, 3] = test_data.iloc[idx, features[node_indices]]

        # The last node is the leaf node - this node doesn't have a decision
        # feature, threshold, or value. We eliminate these.
        tmp_results[-1, 1:] = np.nan

        tmp = pd.DataFrame(tmp_results[:, 1:],
                           index=tmp_results[:, 0].astype(int),
                           columns=columns)
        tmp.index.name = 'node'
        results[test_data.index[idx]] = tmp
            
    return results

def tree_vote_as_cm(clf, tree, test_data, test_labels, tree_state_df):
    predictions = tree.apply(test_data) # Leaf ids
    class_predictions = tree_state_df.loc[predictions, 'node_class']
    cm = confusion_matrix(test_labels, class_predictions, labels=clf.classes_)
    return pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)


def _decision_node_boxplot(plot_data, feature, threshold, gb, ordered_groups,
                           colors, save_fp):
    '''Make a boxplot for decision nodes.
    plot_data : pd.DataFrame
        columns are features
    feature : str
        the feature (column) to plot
    threshold : float
        the decision threshold for this feature at this node
    gb : pd.groupby object
        a groupby object that associates to each key (the members of
        `ordered_groups`) the row indices in `plot_data` that should be
        grouped.
    ordered_groups : iterable
        The order of the groups on the xaxis.
    colors : iterable
        the colors for the boxplot elements for each group. consumed in the
        same order as ordered_groups
    save_fp : str
        filepath to save.
    '''

    fig = plt.figure(figsize=(10,10), dpi=300)
    ax = fig.add_subplot(111)
    # Boxplot
    xlocs = np.arange(len(ordered_groups))
    ylocs = []
    for group in ordered_groups:
        idxs = gb.groups[group]
        tmp = plot_data.loc[idxs, feature].values
        ylocs.append(tmp[~np.isnan(tmp)])

    # For coloring boxplots see here:
    # https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color

    boxps = ax.boxplot(ylocs, positions=xlocs, widths=0.75, notch=True,
                       patch_artist=True)
    items = ['boxes', 'whiskers', 'cap', 'medians', 'caps']
    for x in range(len(ordered_groups)):
        for item in ['boxes', 'medians']:
            plt.setp(boxps[item][x], color=colors[x])
    for x in range(len(ordered_groups)):
        for item in ['fliers']:
            plt.setp(boxps[item][x], markeredgecolor=colors[x])
    for x in range(len(ordered_groups)):
        for item in ['whiskers', 'caps']:
            for idx in [2 * x, 2 * x + 1]:
                plt.setp(boxps[item][idx], color=colors[x])

    ax.hlines(threshold, xlocs[0] - 0.5, xlocs[-1] + 0.5, lw=3,
              linestyle='dotted', zorder=10)

    ax.set_xticks([])
    plt.setp(ax.yaxis.get_ticklabels(), fontsize=40)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.savefig(save_fp, transparent=True)
    plt.close('all')

    ## Old style with filled bars
    # Color per group
    # ylb, yub = plt.ylim()
    # for x in xlocs:
    #     plt.fill_between([x - 0.5, x + 0.5], y1=[ylb, ylb], y2=[yub, yub],
    #                      color=colors[x], alpha=0.2)
    # plt.xlim(xlocs[0] - 0.5, xlocs[-1] + 0.5)
    # plt.ylim(ylb, yub)

def _leaf_node_pieplot(counts, colors, fp, autopct='%1.1f%%', fontsize=40):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111)
    ax.pie(counts, colors=colors, autopct=autopct, startangle=90,
           textprops={'fontsize':fontsize})
    ax.axis('equal')
    fig.savefig(fp, transparent=True)
    plt.close('all')
