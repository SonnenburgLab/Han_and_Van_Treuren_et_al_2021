from collections import defaultdict
from copy import copy

import pandas as pd
import numpy as np

from .util import superset

'''
The purpose of this library is to provide objects which can aid in grouping and
comparing samples across different experiments and runs.

BiologicalSampleGroup - this class holds biologically related samples, most
commonly groups of replicate supernatant samples. The purpose of this class is
to hold all immutable information about a sample (e.g. what culture tube it can
from, when was it run, etc.).

Experiment - this class is for groupings of BiologicalSampleGroup objects.
Instances of this class are meant to aggregate sets of biological replicates
that share important co-variates. For example, the date/time the LCMS machine
was run has a strong influence on the resulting measured data. Thus, if the
supernatants from a microbe were run on different days (i.e. technical
replicates) the measured features and intensities of those features would be
different. By grouping each set of supernatants with the other supernatants
that were run at the respective date/time, the Experiment class allows us to
do batch correction (among other things). The Experiment class is meant to be
flexible and different 'experiments' might be specified by the user to look at
how a particular grouping of samples behaves. For example, imagine that you had
an LCMS run interrupted by a LC leak. After fixing the leak, the second half of
your samples look substantially different in terms of retention time. You will
likely analyze this data with MS-DIAL in two batches - the pre-leak and
post-leak set. Thus, you might create three Experiment classes; one with
pre-leak data, one with post-leak data, and one with both sets of data. You can
compare how the groupings affect downstream processing before settling on one
exact grouping.

The pipeline for going from MS-DIAL analysis to BiologicalSampleGroup's (bsgs)
and Experiment's (exps) is as follows:

1. Using the master sample database and your MS-DIAL analysis files, produce
   `msdata`. This is the set of all `msdata` you have access to (for this
   analysis) and has unique samples names as rows, and feature names as
   columns.

2. Specify the `bsg`'s using the master sample database. This specification
   will likely be programmatic (e.g. a pandas groupby operation). If done on
   master sample database, this will produce a set of `bsg` objects that may
   differ from the data produced in step 1. For example, the master sample
   database may have: (a) samples that were not actually run due to time
   limitations, (b) replicates that contained an error (e.g. 2/3 wells
   looked good, but the third well had a pressure fluctuation or other error),
   (c) the `msdata` built in (1) contains only a small subset of samples of
   interest (compared to all data generated). To resolve these cases, the
   master samples database (`md`) can be subsetted to have only samples found
   in the `msdata`. This guarantees that `bsg` objects will be built only from
   the data as was run on the machine, rather than the whole set of data that
   is recorded in the master sample datbase.

3. Taking the result of (3), specify an experimental design. An experimental
   design means a mapping of which `Experiment` instance will contain
   references to which `bsgs`. This can be done largely programatically. Some
   manual curation will be required at this step. For example, experiments
   which didn't have a media blank will likely need to recieve a synthetic
   media blank or media blank from another experiment.

4. This experimental design (`exp` to `bsgs` mapping) can be saved so that it
   is not necessary to rebuild it. At this step you have a stable
   experimental design that can be reused to make comparisons. It is stable in
   the sense that any updated version of the `msdata` or master sample database
   that does not remove samples will produce the sample results when the
   experimental design is applied.
'''


def squeeze_metadata(series):
    '''Return a single value if metadata is consistent, error otherwise.
    
    Consistent metdata means 1) all nans, 2) all but one nan.
    '''
    if series is None:
        return None
    else:
        tmp = series.isnull()
        if tmp.all():
            return np.nan
        elif (~tmp).sum() == 1 and len(tmp) > 1:
            return series[~tmp].values[0]
        elif len(set(series)) == 1:
            return series.iloc[0] # index doesn't matter - all the same.
        else:
            # There is seemingly inconsistent metadata.
            return None


class BiologicalSampleGroup():

    def __init__(self, samples_metadata):
        '''Init.'''
        # Basic facts about self.
        self.samples = samples_metadata.index.values

        self.sample_type = \
            self._squeeze_metadata(samples_metadata['sample_type'])
        self.clean_by_16s = \
            self._squeeze_metadata(samples_metadata['clean_by_16s'])
        # Data about LC-MS for this BSG.
        self.chromatography = \
            self._squeeze_metadata(samples_metadata['chromatography'])
        self.ionization = \
            self._squeeze_metadata(samples_metadata['ionization'])
        self.run_designation = \
            self._squeeze_metadata(samples_metadata['run_designation'])
        self.lcms_run_date = \
            self._squeeze_metadata(samples_metadata['lcms_run_date'])

        # Data about culturing for this BSG.
        self.media = \
            self._squeeze_metadata(samples_metadata['media'])
        self.preculture_time = \
            self._squeeze_metadata(samples_metadata['preculture_time'])
        self.subculture_time = \
            self._squeeze_metadata(samples_metadata['subculture_time'])
        self.taxonomy = \
            self._squeeze_metadata(samples_metadata['taxonomy'])
        self.culture_source = \
            self._squeeze_metadata(samples_metadata['culture_source'])
        self.matrix_tube = \
            self._squeeze_metadata(samples_metadata['matrix_tube'])

        # If we have to remove samples in data processing, let's remember that.
        self.removed_samples = set()

        # References to other samples.
        self.od_sample_ids = samples_metadata['od_sample_id']


    def __eq__(self, obsg):
        if not isinstance(obsg, self.__class__):
            return False
        else:
            # Must make a check appropriate for each datatype.
            chk1 = set(self.samples) == set(obsg.samples)
            try:
                chk2 = ((self.od_sample_ids == obsg.od_sample_ids) |
                        (self.od_sample_ids.isnull() &
                         obsg.od_sample_ids.isnull())).all()
            except ValueError:
                # Not identically labeled.
                return False

            eq = chk1 & chk2 & (self.removed_samples == obsg.removed_samples)

            attrs = ['chromatography', 'clean_by_16s', 'culture_source',
                     'ionization', 'lcms_run_date', 'matrix_tube',
                     'media', 'preculture_time', 'run_designation',
                     'sample_type', 'subculture_time', 'taxonomy']
            for attr in attrs:
                eq = eq & ((self.__dict__[attr] == obsg.__dict__[attr]) |
                            (pd.isnull(self.__dict__[attr]) &
                             pd.isnull(obsg.__dict__[attr])))
            return eq


    def _squeeze_metadata(self, series):
        result = squeeze_metadata(series)
        if result is None:
            err_str = ('ERROR: Samples `%s` were grouped but appear to have '
                       'inconsistent metadata in field `%s`. They must be '
                       'inspected and added manually. If your grouping is '
                       'generating this error frequently be extremely '
                       'careful as it suggests grouping function is not '
                       'performing correctly.')
            raise ValueError(err_str % (','.join(self.samples), series.name))
        else:
            return result

# Need to check if a single sample as a str (rather than another iterable) has
# been supplied so that we don't iterate over the letters of the string and
# give a non-helpful result.
def get_bsg_with_samples(bsgs, samples):
    '''Find `bsgs` that contains all of `samples` (may be just 1).'''
    count = 0
    for bsg_samples, bsg_obj in bsgs.items():
        if len(set(samples).difference(bsg_samples)) == 0:
            count += 1
            return_samples, return_bsg = bsg_samples, bsg_obj
            if count > 1:
                raise ValueError('`samples` in more than one of `bsgs`.')
    if count == 0:
        raise ValueError('`samples` never found in `bsgs`.')
    else:
        return return_samples, return_bsg

def rebuild_bsg_without_samples(bsg, samples_to_remove):
    '''Rebuild a bsg object after removing specific samples.'''
    remaining_samples = set(bsg.samples).difference(samples_to_remove)
    if len(remaining_samples) == 0:
        # This bsg needs to be removed entirely, no good samples remain.
        return None, None
    else:
        nbsg = copy(bsg)
        nbsg.samples = np.array(list(remaining_samples))
        nbsg.od_sample_ids = bsg.od_sample_ids[np.in1d(bsg.samples,
                                                       nbsg.samples)]
        # Update removed samples, if we've done multiple rounds of removal
        # we want to keep track of that.
        rs = set(bsg.samples).difference(nbsg.samples)
        nbsg.removed_samples = rs.union(nbsg.removed_samples)

        return frozenset(nbsg.samples), nbsg


class Experiment():

    def __init__(self, culture_sample_groups, unspent_medias, media_in_samples,
                 cultured_in, ni_blank_samples, qc_samples, md,
                 other_samples=None):

        # These are lists of frozensets; keys to the bsg objects
        self.culture_sample_groups = culture_sample_groups

        # Media maps and samples
        self.unspent_medias = unspent_medias
        self.media_in_samples = media_in_samples
        self.cultured_in = cultured_in

        # Check for needed media
        self.has_needed_unspent_media = self._check_has_needed_unspent_media()

        # No injection blanks.
        self.ni_blanks = ni_blank_samples

        # QC samples.
        self.qc_samples = qc_samples

        # Samples that were part of the groupby operation but were not included
        # in any bsg or as no injection blanks. Only istd_blanks and qcs.
        self.other_samples = other_samples

        # Samples
        self.samples = {}
        self.samples['supernatants'] = superset(culture_sample_groups)
        self.samples['unspent_medias'] = superset(unspent_medias.values())
        self.samples['ni_blanks'] = set(ni_blank_samples)
        self.samples['qcs'] = set(qc_samples)
        self.samples['all'] = superset(self.samples.values())

        # Information shared by all samples - this can help in e.g. adding
        # media blanks from one experiment to another experiment.
        self.culture_date = \
            squeeze_metadata(md.loc[self.samples['supernatants'],
                                    'preculture_date'])
        self.lcms_run_date = squeeze_metadata(md.loc[self.samples['all'],
                                                     'lcms_run_date'])
        self.ionization = squeeze_metadata(md.loc[self.samples['all'],
                                           'ionization'])
        self.chromatography = squeeze_metadata(md.loc[self.samples['all'],
                                                      'chromatography'])
        self.exp_id = squeeze_metadata(md.loc[self.samples['all'],
                                              'experiment'])

        # Keep a record of samples that have been removed after a rebuild.
        self.removed_samples = set()


    def _check_has_needed_unspent_media(self):
        unspent = set(self.unspent_medias.keys())
        spent = set(self.media_in_samples.values())

        if unspent.issuperset(spent):
            return True
        else:
            return False


    def add_unspent_media_samples(self, unspent_medias_to_add, force=False):
        '''Add all k:v media:set(samples) to media, update if we have all.'''
        # Check if we are trying to overwrite existing medias.
        to_add = set(unspent_medias_to_add.keys())
        existing = set(self.unspent_medias.keys())
        if len(to_add.intersection(existing)) > 0 and not force:
            raise ValueError('Trying to add an unspent media sample to '
                             'experiment that already has that media '
                             'won\'t do this without `force`.')
        else:
            for media, samples in unspent_medias_to_add.items():
                self.unspent_medias[media] = samples

        # Update if we now have the needed media.
        self.has_needed_unspent_media = self._check_has_needed_unspent_media

        # Update the samples information
        self.samples['unspent_medias'] = superset(self.unspent_medias.values())
        self.samples['all'] = superset(self.samples.values())


    def has_sample(self, sid):
        return (sid in samples['all'])


    def __eq__(self, oexp):
        if not isinstance(oexp, self.__class__):
            return False
        else:
            # These are lists of sets whose order may be different. Check them
            # separately.
            skip = set(['culture_sample_groups', 'cultured_in',
                        'media_in_samples'])

            chk = self.__dict__.keys() == oexp.__dict__.keys()
            for k in self.__dict__.keys() - skip:
                chk = chk & (self.__dict__[k] == oexp.__dict__[k])

            for k in skip:
                chk = chk & (set(self.__dict__[k]) == set(oexp.__dict__[k]))
            return chk


# def _remove_from_frozenset(set1, set2):
#     '''Remove `set2` from `set1`; return None if all removed.'''
#     to_remove = set1.intersection(set2)
#     if len(to_remove) == 0:
#         # Don't remove any.
#         r = set1
#     elif (len(to_remove) > 0) and (len(to_remove) < len(set1)):
#         # Remove some.
#         r =frozenset(set1.difference(to_remove))
#     elif len(to_remove) == len(sg):
#         # Need to completely remove this.
#         r = None
#     return r


# def rebuild_exp_without_samples(exp, to_remove, bsgs, md):
#     samples_to_keep = _remove_from_frozenset(exp.samples['all'], to_remove)
#     if samples_to_keep is None:
#         raise ValueError('All samples removed.')
#     tmp = partition_samples(samples_to_keep, bsgs, md)
#     n_exp = Experiment(culture_sample_groups=tmp[0],
#                        unspent_medias=tmp[1],
#                        media_in_samples=tmp[2],
#                        cultured_in=tmp[3],
#                        ni_blank_samples=tmp[4],
#                        md=md)
#     n_exp.removed_samples = exp.samples['all'].difference(n_exp.samples['all'])
#     return n_exp

def partition_samples(exp_samples, bsgs, md):
    '''Partition `exp_samples` into groups to build an `Experiment` object.

    Extended Summary
    ----------------
    This function takes a set of samples which were all generated in one
    experiment and partitions them into different groups. The groups correspond
    to culture samples (supernatants), unspent media samples, no injection
    blanks, and qcs. The `bsgs` are normally going to contain a superset of
    the samples passed in `exp_samples`. We check through each bsg to see if
    all it samples are in the `exp_samples` and then use it's samples if so.

    Parameters
    ----------
    exp_samples : list, np.array
        Samples from an experiment. Generally these will be all the samples
        from an experiment that have not been eliminated by upstream quality
        control.
    bsgs : dict
        The dictionary of frozenset:bsg that defines the bsg groupings of all
        samples in all experiments. Note that `bsgs` will contain many objects
        which do not have samples listed in `exp_samples`.
    md : pd.DataFrame
        The dataframe containing the metadata. Must at least contain every
        sample found in `exp_samples`.
    '''
    culture_sample_groups = []
    unspent_medias = {}
    media_in_samples = {}
    cultured_in = defaultdict(list)

    for bsg_key, bsg in bsgs.items():
        if all((s in exp_samples for s in bsg_key)):
            # This is the only thing that must be in the bsgs to build an
            # experiment.
            if bsg.sample_type == 'supernatant':
                culture_sample_groups.append(bsg_key)
                media_in_samples[bsg_key] = bsg.media
            # These may be in the bsgs.
            elif bsg.sample_type == 'media_blank':
                unspent_medias[bsg.media] = bsg_key
            # If these were built into bsgs we can ignore them and add them as
            # lists below.
            elif bsg.sample_type in ('qc', 'ni_blank', 'istd_blank'):
                pass
            else:
                raise ValueError('Unknown sample type for this BSG.')
        else:
            # The samples from this bsg were not in the samples designated as
            # part of this experiment by the user. We won't add them.
            pass

    # Attempt to build a media to culture map.
    for culture_sample_group, media in media_in_samples.items():
        cultured_in[media].append(culture_sample_group)

    # Now, partition 'ni_blank' samples
    tmp = (md.loc[exp_samples, 'sample_type'] == 'ni_blank')
    ni_blank_samples = set(tmp.index[tmp.values])

    # Now, add 'qc' samples
    tmp = (md.loc[exp_samples, 'sample_type'] == 'qc')
    qc_samples = set(tmp.index[tmp.values])

    # Now, add 'istd_blank' samples
    tmp = (md['sample_type'] == 'istd_blank')
    ib_samples = set(tmp.index[tmp.values])

    # If we haven't partitioned all the samples of this experiment into the
    # appropriate groups it means that something has gone wrong. Likely, the
    # `bsgs` didn't contain all the groups of interest. We need to fail to
    # indicate that some grouping operation is not right.
    tmp = ni_blank_samples.union(qc_samples).union(ib_samples)
    tmp = tmp.union(superset(culture_sample_groups))
    tmp = tmp.union(superset(cultured_in.values()))
    tmp = tmp.union(superset(unspent_medias.values()))
    samples_remaining = set(exp_samples) - tmp
    if len(samples_remaining) > 0:
        raise ValueError(('Some elements of `exp_samples` (%s) were not '
                          'grouped into categories. This likely indicates an '
                          'error in `bsgs` construction.')
                          % ','.join(samples_remaining))

    return (culture_sample_groups, unspent_medias, media_in_samples,
            cultured_in, ni_blank_samples, qc_samples)





