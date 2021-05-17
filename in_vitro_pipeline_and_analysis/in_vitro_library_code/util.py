import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform


def superset(list_of_sets, starting_set=set()):
    if isinstance(list_of_sets, str) or isinstance(starting_set, str):
        raise ValueError('Won\'t iterate over a string. Likely not the set '
                         'you want.')
    superset = starting_set
    for s in list_of_sets:
        superset = superset.union(s)
    return superset


def intra_replicate_correlation(data, samples, as_squareform=True):
    n = len(samples)
    tmp = data.loc[samples, data.loc[samples].notnull().all(0).values]
    if n > 1:
        c = np.corrcoef(tmp.values)[np.triu_indices(n, 1)]
    else:
        c = np.array([1.])
    if as_squareform:
        return squareform(c) # Note the diagonal will be 0 rather than 1.
    else:
        return c

def intra_experiment_correlation(data, samples_by_exps):
    pass

def fc_and_delta(supernatant_samples, media_samples, msdata):
    '''Return 3 dfs - summary, log2 fc, lof2 delta.'''
    tmp_sn = msdata.loc[supernatant_samples, :]
    sn_detected_idxs = tmp_sn.notnull().all(0).values

    tmp_ms = msdata.loc[media_samples, :]
    ms_detected_idxs = tmp_ms.notnull().all(0).values

    # Check to make sure media blanks and supernatant blanks have same detected
    # metabolites. They might be different if different MS-DIAL runs or diff
    # experiments.
    joint_idx = sn_detected_idxs & ms_detected_idxs
    if (joint_idx != sn_detected_idxs).any():
        print('Warning: different non-nan indices between media blanks and '
              'supernatant samples.')
    sn = tmp_sn.loc[:, joint_idx]
    ms = tmp_ms.loc[:, joint_idx]

    ms_mean_log2 = np.log2(1 + ms).mean(0)
    sn_log2 = np.log2(1. + sn)

    fc = sn_log2 - ms_mean_log2
    delta = np.log2(1 + np.abs((sn - ms.mean(0))))

    tmp = np.vstack([fc.values.mean(0), delta.values.mean(0),
                     ms_mean_log2.values]).T

    cols = ['b/m', 'b-m', 'media_mean']
    summary_df = pd.DataFrame(tmp, index=fc.columns,
                              columns=cols).sort_values('b/m')
    # Order dfs by the summary_df
    return (summary_df, fc.T.loc[summary_df.index, :],
            delta.T.loc[summary_df.index, :])


def media_based_transform(data, sample_map, fc_or_delta='fc',
                          data_is_log_transformed=True):
    '''Normalize all samples in `data` using `sample_map`. 

    Warning
    -------
    This function expects that the data has the same columns for the
    normalizing and samples to be normalized. This is normally the case, but
    there are situations where it might not occur. The most common example is
    if the normalizing samples come from a different experiment than the
    samples to be normalized. In this case, there are likely non-overlapping
    detections. The samples that are in the symmetric difference of the two
    sets (e.g. metabolites in the normalizing samples only, and samples in the
    to be normalized samples only) will have nan values after this transform.

    Extended Summary
    ----------------
    This function transforms count data into fold-change or delta data. The
    transform is calculated for each metabolite (column) independently. For
    each (key, value) pair in `sample_map`, the key is the set of samples to be
    normalized, and the value is the set of samples that will be usef for the 
    normalization. The mean of the values of the normalizing samples will be
    used. For example:

    sample_map = {frozenset(['s1', 's2', 's3']): frozenset(['s4', 's5'])

    This sample_map would indicate that samples 1-3 should be normalized by
    samples 4 and 5. The average of s4 and s5 would be subtracted (if 'fc' was
    requested and data was already log transformed) from each of samples 1-3,
    for each metabolite independently.

    Multiple samples can be normalized by the same samples (e.g. you can use a
    media blank to normalize many different samples), but the function will
    fail if you attempt to normalize the same sample by two different groups
    of samples.

    Paramters
    ---------
    data : pd.DataFrame
        Rows are sample IDs, columns are metabolites. Data is count of features
        (possibly) log transformed.
    sample_map : dict
        key:value pairs are each frozensets of samples. The key is the set of
        samples to be normalized by the mean of the set of samples in the
        value.
    '''
    if fc_or_delta=='delta' and data_is_log_transformed:
        raise ValueError('Won\'t compute delta on log transformed data.')

    samples = superset(sample_map.values()).union(superset(sample_map.keys()))
    if len(samples.symmetric_difference(data.index)) > 0:
        raise ValueError('`data` index and samples in `sample_map` must have '
                         'the same membership.')

    # Test to make sure we aren't trying to normalize the same sample in
    # different ways.
    c = 0
    for i in sample_map.keys():
        c+=len(i)
    if len(superset(sample_map.keys())) != c:
        raise ValueError('`sample_map` assigns at least one sample to be '
                         'normalized by at least two different samples. This '
                         'will result in the transform perfoming '
                         'unpredictably (based on the order of iteration of '
                         'dict). This also indicates an error in grouping '
                         'logic.')

    # df.loc calls are slow. Build a dict of `data` row indexes to place data
    # in the appropriate row.
    row_idx_map = {sample:row for row, sample in enumerate(data.index)}

    result = np.zeros(data.shape, dtype=float)

    for to_norm, norm_by in sample_map.items():
        to_norm_rows = [row_idx_map[i] for i in to_norm]
        norm_by_rows = [row_idx_map[i] for i in norm_by]

        norm_vals = data.iloc[norm_by_rows].mean(0).values

        if fc_or_delta == 'fc':
            if data_is_log_transformed:
                result[to_norm_rows] = (data.iloc[to_norm_rows].values -
                                        norm_vals)
            else:
                result[to_norm_rows] = (data.iloc[to_norm_rows].values /
                                        norm_vals)
        elif fc_or_delta == 'delta':
            result[to_norm_rows] = data.iloc[to_norm_rows].values - norm_vals

    return pd.DataFrame(result, index=data.index, columns=data.columns)


def istd_tic_transform(data, istds, by='median'):
    istd_tic = data.loc[:, istds].sum(1)
    if istd_tic.isnull().any():
        raise ValueError('One or more of `istds` has a `nan` value. '
                         'Can\'t transform.')
    if by == 'median':
        nd = data.multiply((istd_tic.median() / istd_tic).values, axis='rows')
    elif by == 'mean':
        nd = data.multiply((istd_tic.mean() / istd_tic).values, axis='rows')
    else:
        raise ValueError('Unknown method for transform.')
    # If istd_tic is 0 for any samples (some blanks show this), nd will have
    # nans or infs. These should remain as 0's; they are true detects (with 0
    # intensity) and thus no adjustment should be necessary. Our model assumes
    # that there is no additive error modeled by the ISTDs.
    samples = istd_tic.index[(istd_tic == 0)]
    nd.loc[samples, :] = data.loc[samples, :]
    return nd


def nearest_istd_transform(data, feature_data, istds):
    pass

def consensus_istd_transform(data, istds):
    pass


def join_across_modes(sample_link, c18p_data, c18n_data, hilicp_data):
    # If all three modes are missing, drop the row. If at least one sample/mode
    # is available, keep it.
    nc1 = c18p_data.shape[1]
    nc2 = c18n_data.shape[1]
    nc3 = hilicp_data.shape[1]
    ncols = nc1 + nc2 + nc3
    cols = np.concatenate((c18p_data.columns.values, c18n_data.columns.values,
                           hilicp_data.columns.values))

    ndata = np.zeros((sample_link.shape[0], ncols), dtype=float) * np.nan

    modes = ('c18positive', 'c18negative', 'hilicpositive')

    for mode, data, bounds in zip(modes,
                                  (c18p_data, c18n_data, hilicp_data),
                                  ((0, nc1), (nc1, nc1+nc2), (nc1+nc2,ncols))):
        # np.in1d won't work because sl has mixed type; nans and strs.
        idxs = np.empty(sample_link.shape[0], dtype='bool')
        for i, sample in enumerate(sample_link[mode]):
            if pd.notnull(sample) and sample in data.index.values:
                idxs[i] = True
            else:
                idxs[i] = False

        ndata[idxs, bounds[0]:bounds[1]] = \
            data.loc[sample_link.loc[idxs, mode].values, :]

    df = pd.DataFrame(ndata, index=sample_link.index, columns=cols)
    rows_to_remove = df.isnull().all(1).values

    return df.loc[~rows_to_remove, :], sample_link.loc[~rows_to_remove, :]


def join_msdialdf_sampledb(msdial_df, sdb, selectors, additional_columns=None,
                           metabolite_translation_dict=None):
    '''Take an MS-DIAL result and combine it with sample database.

    Extended Summary:
    '''
    idxs = True
    for field, value in selectors.items():
        idxs = idxs & (sdb[field] == value)

    # Produce {'A08_r3': 's01425': , ...}
    misses = []
    sid_map = {}
    for sdb_sid, msdial_sid in sdb.loc[idxs,
                                       'ms_dial_sample_name'].iteritems():
        if msdial_sid in msdial_df.columns:
            sid_map[msdial_sid] = sdb_sid
        else:
            misses.append(msdial_sid)

    # First, isolate the msdata that we want for this MS-DIAL df.
    data = msdial_df.loc[:, sid_map.keys()].values
    idx = msdial_df['Metabolite name'].values
    columns = sid_map.values()
    msdata = pd.DataFrame(data, index=idx, columns=columns).T

    # Second, isolate the additional feature data we are intersted in.
    data = msdial_df[additional_columns].values
    feature_data = pd.DataFrame(data, index=idx, columns=additional_columns)

    # If a translation table for metabolite names is supplied, apply it here.
    if metabolite_translation_dict is not None:
        if set(msdata.columns).issubset(metabolite_translation_dict.keys()):
            msdata.columns = [metabolite_translation_dict[i] for i in
                              msdata.columns]
            feature_data.index = [metabolite_translation_dict[i] for i in
                                  feature_data.index]
        else:
            q = set(msdata.columns).difference(metabolite_translation_dict.keys())
            raise ValueError('The `metabolite_translation_dict` supplied does '
                             'not contain a translation for all metabolites. '
                             'missing: %s' % ','.join(q))

    return (msdata, feature_data, misses)


def name_detection_id(d_id, chemical_info, condense=True):
    '''Determine the name of a detection according to hardcoded rules.

    Parameters
    ----------
    d_id : str
        The detection id.
    chemical_info : pd.DataFrame
        The chemical information dataframe.
    condense : bool
        If True (default) then reports only unique compound id's in final
        string. Helpful when the library lists several co-eluting
        stereoisomers.
    '''

    tmp = chemical_info.loc[(chemical_info['dname'] == d_id),
                            ['Compound', 'Peak']]

    name = []
    for _, row in tmp.iterrows():
        if pd.notnull(row[1]):
            n = row[0] + '_' + row[1]
        else:
            n = row[0]
        name.append(n)

    if condense:
        name = sorted(set(name))
    else:
        name = sorted(name)

    return ','.join(name)



def add_multipeaks(msdata, to_combine, new_names):
    '''Add counts of specified features together.

    Extended Summary
    ----------------
    This function is intended to be used on raw count matrices. Features that
    represent the same chemical compound are added together. If one of the
    peaks is a non-detect (`np.nan`) it is ignored in the addition. The
    individual features that are added together are dropped from the table and
    their summed value named by the corresponding value in `new_names`.

    Parameters
    ----------
    msdata : pd.DataFrame
        Rows are samples, columns are features, data are feature counts.
    to_combine : list
        Each entry is a list containing features that should be added. Entries
        can't be tuples (must be lists) or pandas will interpret them as
        multiindex keys.
    new_names : list
        Each entry is the new name of features that are added. The ith entry
        of this list is the new name of the sum of the ith tuple in the
        `to_combine` list.

    Examples
    --------
    tmp = np.arange(40, dtype=float).reshape(5, 8)
    index = ['s%s' % i for i in range(5)]
    columns = ['f%s' % i for i in range(8)]
    data = pd.DataFrame(tmp, index=index, columns=columns)

    to_combine = [['f0', 'f1']]
    new_names = ['f_1+2']
    obs = combine_multipeaks(data, to_combine, new_names)
    #       f2    f3    f4    f5    f6    f7  f_0+1
    # s0   2.0   3.0   4.0   5.0   6.0   7.0    1.0
    # s1  10.0  11.0  12.0  13.0  14.0  15.0   17.0
    # s2  18.0  19.0  20.0  21.0  22.0  23.0   33.0
    # s3  26.0  27.0  28.0  29.0  30.0  31.0   49.0
    # s4  34.0  35.0  36.0  37.0  38.0  39.0   65.0
    '''
    to_drop = superset(to_combine)

    if len(to_drop.difference(msdata.columns)) > 0:
        raise ValueError('Some features to be combined not found in `msdata`.')
    # If the user passes feature names that are not going to be eliminated, we
    # need to fail to avoid silent data overwrites.
    if len((set(msdata.columns) - to_drop).intersection(new_names)) > 0:
        raise ValueError('The new feature names have overlap with existing '
                         'feature names that are not going to be dropped '
                         'by this function.')
    
    tmp = []
    for (c1, c2) in to_combine:
        tmp.append(msdata[c1] + msdata[c2])

    tmp_data = msdata.drop(to_drop, axis='columns')
    for col_name, values in zip(new_names, tmp):
        tmp_data[col_name] = values

    return tmp_data


def combine_features_in_fc_data(fc_data, features_to_group, new_names,
                                pref_strength_threshhold=3):
    '''Combine fold change values for feature groups in `to_combine`.

    Extended Summary
    ----------------
    This function deals with situations where a chemical compound is detected
    in multiple modes in on sample. In some cases, we want to select just one
    mode for a compound, and in other cases we want to average all modes. The
    logic is as follows:
        If a feature group has a preferred feature (or features), and the
        preference for that preferred feature is high enough, that feature
        (or the average of the preferred features) will be used in place of the
        the values of the features in the feature group and they will be
        dropped.
        If a feature group has a preferred feature (or features), but the
        preference for replacement is low enough, then the average of all of
        the features in the feature group will be used.

    Parameters
    ----------
    fc_data : pd.DataFrame
        Rows are samples, columns are detected features. Entries are fold
        change values.
    features_to_group : list
        Each entry is a triple containing the features to be combined, the
        preferred feature(s), and the preference strength.
    pref_strength_threshhold : int
        The score at which a feature will be considered preferred enough to be
        used instead of the average of all existing features.

    Example
    -------
    tmp = np.arange(40, dtype=float).reshape(5, 8)
    index = ['s%s' % i for i in range(5)]
    columns = ['f%s' % i for i in range(8)]
    data = pd.DataFrame(tmp, index=index, columns=columns)

    features_to_group = [(['f0', 'f1'], ['f1'], 3),
                         (['f3', 'f2'], ['f2'], 2),
                         (['f4', 'f5', 'f7'], ['f4', 'f5'], 3)]
    new_names = ['f1_1', 'f2_1', 'f3_1']
    pref_strength_threshhold = 3
    obs = combine_features_in_fc_data(data, features_to_group,
                                      new_names, pref_strength_threshhold)
    #       f6  f1_1  f2_1  f3_1
    # s0   6.0   1.0   2.5   4.5
    # s1  14.0   9.0  10.5  12.5
    # s2  22.0  17.0  18.5  20.5
    # s3  30.0  25.0  26.5  28.5
    # s4  38.0  33.0  34.5  36.5
    '''
    tmp1 = []
    tmp2 = []
    tmp3 = []
    for row in features_to_group:
        tmp1.extend(row[0])
        tmp2.extend(row[1])
        for i in row[1]:
            tmp3.append(i in row[0])

    if len(tmp1) != len(set(tmp1)):
        print(tmp1)
        raise ValueError('A feature has been assigned to more than one group.')
    if len(tmp2) != len(set(tmp2)):
        raise ValueError('A preferred feature has been assigned to more than '
                         'one group.')
    if not all(tmp3):
        raise ValueError('At least one preferred feature is not a member of '
                         'the features of its to be combined group.')
    # If the user passes feature names that are not going to be eliminated, we
    # need to fail to avoid silent data overwrites.
    if len((set(fc_data.columns) - set(tmp1)).intersection(new_names)) > 0:
        raise ValueError('The new feature names have overlap with existing '
                         'feature names that are not going to be dropped '
                         'by this function.')

    if len(new_names) != len(set(new_names)):
        raise ValueError('Multiple feature groups are being renamed to the '
                         'same new name.')

    to_drop = []
    new_data = []
    for (fs, pf, ps) in features_to_group:
        
        ### To resolve cases where features are missing.
        fs_idxs = np.in1d(fs, fc_data.columns)
        pf_idxs = np.in1d(pf, fc_data.columns)
        
        if fs_idxs.sum() < fs_idxs.shape[0]:
            # Some feature not found
            print('Feature `%s` not detected.' % 
                  ','.join(np.array(fs)[~fs_idxs]))
        if pf_idxs.sum() < pf_idxs.shape[0]:
            # Some feature not found
            print('Preferred feature `%s` not detected.' % 
                  ','.join(np.array(pf)[~pf_idxs]))
            print()

        # Go through existing features
        fs = np.array(fs)[fs_idxs]
        pf = np.array(pf)[pf_idxs]
        ####
        
        if ps < pref_strength_threshhold:
            # We want to average rather than select the best feature.
            # pandas mean will ignore nan entries safely
            new_data.append(fc_data[fs].mean(1))
        else:
            # Want want to select the best feature(s) where it exists,
            # otherwise average.
            tmp = fc_data[pf].mean(1)
            idxs = pd.isnull(tmp).values
            tmp[idxs] = fc_data[fs].mean(1)[idxs]
            new_data.append(tmp)
        to_drop.extend(fs)

    tmp_data = fc_data.drop(to_drop, axis='columns')

    for col_name, values in zip(new_names, new_data):
        tmp_data[col_name] = values

    return tmp_data