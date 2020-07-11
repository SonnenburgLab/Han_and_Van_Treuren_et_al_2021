import numpy as np
import pandas as pd
from collections import defaultdict
import os

class MouseDataAnalysis:
    """
    Generates raw ion count, ISTD-corrected, and fold-change matrices, by combining the mouse
    sample database and the MS-DIAL output files from each experiments across all three analytical
    methods (referred to as 'modes' below).


    Attributes
    ----------
    non_community_db : pd.DataFrame
        The non-community (mono-colonization and conventional) portion of the mouse sample database.
    community_db : pd.DataFrame
        The community portion of the mouse sample database

        The mouse sample database has the following key columns:

        sample_id: identifies a single sample

        run_id: identifies a single run. 3 runs are performed on each sample for the 3 modes:
            C18 positive, C18 negative, HILIC positive

        ms_dial_sample_name: identifies a single sample in the MSDIAL analysis worksheets

        chromatography and ionization: the mode setting for the run

        sample_type: the type of sample (caecal, urine, etc...)

        colonization: the bacteria present in the sample

        experiment: one of mono-colonization, community, or conventional

    msdial_analysis_map : pd.DataFrame
        Map pointing to all MSDIAL analysis files (must be in the same directory)

        Each row points to a single worksheet in a file containing the MSDIAL analysis
        for a particular experiment, mode, and sample type
    msdial_analysis_dir : str
        Path to where the MSDIAL analysis map file lives
    cpd_library : pd.DataFrame
        Compound library containing dname to compound name mappings

        This is needed to figure out the actual metabolite / compound names corresponding to the
        dnames (e.g. m_c18n_0001 => N-ACETYLTRYPTOPHAN)
    """


    ISTD_CHOICES = {
        'c18positive' : [
            'IS_2-FLUROPHENYLGLYCINE',
            'IS_4-BROMO-PHENYLALANINE',
            'IS_4-CHLORO-PHENYLALANINE',
            'IS_LEUCINE-5,5,5-D3',
            'IS_METHIONINE-METHYL-D3',
            'IS_N-BENZOYL-D5-GLYCINE',
            'IS_INDOLE-2,4,5,6,7-D5-3-ACETIC ACID',
            'IS_PHENYLALANINE-2,3,4,5,6-D5',
            'IS_TRYPTOPHAN-2,4,5,6,7-D5',
            'IS_PROGESTERONE-D9',
            'IS_D15-OCTANOIC ACID',
            'IS_D19-DECANOIC ACID',
            'IS_D27-TETRADECANOIC ACID',
            'IS_TRIDECANOIC ACID'
        ],
        'c18negative' : [
            'IS_4-BROMO-PHENYLALANINE',
            'IS_4-CHLORO-PHENYLALANINE',
            'IS_LEUCINE-5,5,5-D3',
            'IS_N-BENZOYL-D5-GLYCINE',
            'IS_INDOLE-2,4,5,6,7-D5-3-ACETIC ACID',
            'IS_PHENYLALANINE-2,3,4,5,6-D5',
            'IS_TRYPTOPHAN-2,4,5,6,7-D5',
            'IS_GLUCOSE-1,2,3,4,5,6,6-D7',
            'IS_D15-OCTANOIC ACID',
            'IS_D19-DECANOIC ACID',
            'IS_D27-TETRADECANOIC ACID',
            'IS_TRIDECANOIC ACID'
        ],
        'hilicpositive' : [
            'IS_4-BROMO-PHENYLALANINE',
            'IS_4-CHLORO-PHENYLALANINE',
            'IS_LEUCINE-5,5,5-D3',
            'IS_METHIONINE-METHYL-D3',
            'IS_INDOLE-2,4,5,6,7-D5-3-ACETIC ACID',
            'IS_PHENYLALANINE-2,3,4,5,6-D5',
            'IS_TRYPTOPHAN-2,4,5,6,7-D5',
            'IS_PROGESTERONE-D9'
        ]
    }

    '''
    Based on literature searches, the following metabolites (referred to by their dnames from the reference library)
    were removed from the final matrices, because they were unlikely to be naturally produced in mice.
    '''
    METABOLITES_TO_REMOVE = [
        # AMILORIDE
        'm_c18p_0388',
        # ATENOLOL
        'm_hilicp_0257',
        # BIS(2-ETHYLHEXYL)PHTHALATE
        'm_c18p_0521',
        # DILTIAZEM
        'm_c18p_0530',
        # ETOMIDATE
        'm_hilicp_0244',
        # FORMONONETIN
        'm_c18n_0427',
        'm_c18p_0429',
        'm_hilicp_0261',
        # ISOEUGENOL
        'm_hilicp_0144',
        # METFORMIN
        'm_c18p_0108',
        # PILOCARPINE
        'm_c18p_0364',
        # CHLORPROMAZINE
        'm_hilicp_0288',
        # CIMETIDINE
        'm_c18p_0410'
    ]


    def __init__(self,
                 non_community_db=None,
                 community_db=None,
                 msdial_analysis_map=None,
                 msdial_analysis_dir=None,
                 cpd_library=None):
        self.non_community_db = non_community_db
        self.community_db = community_db
        self.msdial_analysis_map = msdial_analysis_map
        self.msdial_analysis_dir = msdial_analysis_dir
        self.cpd_library = cpd_library

    def rename_matrix(self, matrix, dname_cpd_map):
        return matrix.rename(columns=dname_cpd_map).sort_index(axis=1)

    def get_mode_from_dname(self, dname):
        if '_c18p_' in dname:
            return 'c18positive'
        elif '_c18n_' in dname:
            return 'c18negative'
        elif '_hilicp_' in dname:
            return 'hilicpositive'


    def sum_peaks(self, raw_ion_counts_matrix, dname_cpd_map):
        """
        For dname columns that correspond to the same compound (Peak 1 and 2),
        combine them into a single column with the summed raw ion counts
        """

        dnames_by_cpd = defaultdict(list)

        for dname in list(filter(lambda dname: dname in dname_cpd_map, raw_ion_counts_matrix.columns)):
            cpd = dname_cpd_map[dname]
            dnames_by_cpd[cpd].append(dname)

        for cpd, dnames in dnames_by_cpd.items():
            if len(dnames) == 1:
                continue

            #print(f'Summing peaks for {cpd} with dnames {dnames}')

            # Place the summed raw ion counts under the first dname column
            raw_ion_counts_matrix[dnames[0]] = raw_ion_counts_matrix[dnames] \
                .apply(lambda vals: np.nan if np.isnan(vals).all() else np.sum(vals), axis=1)

            # Get rid of the other dname column
            raw_ion_counts_matrix = raw_ion_counts_matrix.drop(columns=dnames[1:])

        return raw_ion_counts_matrix


    def join_msdialdf_sampledb(self, msdial_df, mouse_sample_db, exp_selection):
        # Take an MS-DIAL result and combine it with sample database.

        idxs = True
        for field, value in exp_selection.items():
            idxs = idxs & (mouse_sample_db[field] == value)

        misses = []

        # Construct map of ms_dial_sample_name values to run_id values
        sid_map = {}
        for run_id, msdial_sample_id in mouse_sample_db.loc[idxs, 'ms_dial_sample_name'].iteritems():
            if msdial_sample_id in msdial_df.columns:
                sid_map[msdial_sample_id] = run_id
            else:
                misses.append(msdial_sample_id)

        # Exclude metabolites that have been removed currently by denotation of 'x' next to the metabolite name
        if 'Remove' in msdial_df.columns:
            msdial_df = msdial_df[msdial_df['Remove'] != 'x']

        # Isolate the msdata that we want for this MS-DIAL df.
        data = msdial_df.loc[:, sid_map.keys()].values

        # The "Metabolite name" column holds the metabolite dnames
        idx = msdial_df['Metabolite name'].values
        columns = sid_map.values()

        # Get the transpose of the data so that the metabolites become columns
        msdata = pd.DataFrame(data, index=idx, columns=columns).T

        return (msdata, misses)

    def read_msdial_analyses(self, sample_db, map_df, chromatography, ionization):
        """
        Read in the MSDIAL analysis worksheets specific to the current mode
        """

        filtered_runs = map_df[(map_df.chromatography == chromatography) & \
            (map_df.ionization == ionization)]

        msdata_dfs = []
        misses = []

        for idx, run in filtered_runs.iterrows():
            exp = run['experiment']
            sample_type = run['sample_type']
            chromatography = run['chromatography']
            ionization = run['ionization']

            full_filepath = os.path.join(self.msdial_analysis_dir, run['msdial_fp'])

            #print()
            #print('starting')
            #print(f"full_filepath={full_filepath}")
            #print(f"sheetname={run['sheetname']}")
            #print(f"sample_type={run['sample_type']}")
            #print(f"chromatography={run['chromatography']}")
            #print(f"ionization={run['ionization']}")

            exp_selection = {
                'experiment': exp,
                'sample_type': sample_type,
                'chromatography': chromatography,
                'ionization': ionization
            }

            cur_msdata, cur_misses = \
                self.join_msdialdf_sampledb(pd.read_excel(io=full_filepath, sheet_name=run['sheetname']),
                                            sample_db,
                                            exp_selection)

            msdata_dfs.append(cur_msdata)
            misses += cur_misses

        return (pd.concat(msdata_dfs, sort=True), misses)


    def get_non_community_matrix(self):
        """
        Read in non-community data (mono-colonization and conventional) from MSDIAL analysis files
        """

        non_community_map = self.msdial_analysis_map[self.msdial_analysis_map['experiment'] != 'community']

        c18pos_msdata, c18pos_misses = self.read_msdial_analyses(sample_db=self.non_community_db,
                                                                 map_df=non_community_map,
                                                                 chromatography='c18',
                                                                 ionization='positive')

        c18neg_msdata, c18neg_misses = self.read_msdial_analyses(sample_db=self.non_community_db,
                                                                 map_df=non_community_map,
                                                                 chromatography='c18',
                                                                 ionization='negative')

        hilicpos_msdata, hilicpos_misses = self.read_msdial_analyses(sample_db=self.non_community_db,
                                                                     map_df=non_community_map,
                                                                     chromatography='hilic',
                                                                     ionization='positive')

        non_community_db = self.non_community_db.copy(deep=True)
        non_community_db['mode'] = self.non_community_db['chromatography'] + self.non_community_db['ionization']

        joined_data = c18pos_msdata \
            .join(c18neg_msdata, how='outer') \
            .join(hilicpos_msdata, how='outer') \
            .join(non_community_db[['experiment', 'sample_type', 'colonization', 'sample_id']], how='inner')

        joined_data.index.name='run_id'

        # Get one combined sample data row for each set of runs
        # (c18pos, c18neg, hilicpos from the same sample)
        sample_data = joined_data \
            .groupby(['sample_id']) \
            .first() \
            .join(joined_data
                .join(non_community_db[['mode']])
                .reset_index()
                # Pivot so that we know which run ids are associated with each sample
                .pivot(index='sample_id', columns='mode', values='run_id')
            )

        metadata_columns = [
            'experiment',
            'sample_type',
            'colonization',
            'c18positive',
            'c18negative',
            'hilicpositive'
        ]

        metadata = sample_data[metadata_columns]
        sample_data = sample_data.drop(columns=metadata_columns)

        return (sample_data, metadata)


    def get_community_matrix(self):
        """
        Read in community data from MSDIAL analysis files
        """

        community_map = self.msdial_analysis_map[self.msdial_analysis_map['experiment'] == 'community']

        c18pos_msdata, c18pos_misses = self.read_msdial_analyses(sample_db=self.community_db,
                                                                 map_df=community_map,
                                                                 chromatography='c18',
                                                                 ionization='positive')

        c18neg_msdata, c18neg_misses = self.read_msdial_analyses(sample_db=self.community_db,
                                                            map_df=community_map,
                                                            chromatography='c18',
                                                            ionization='negative')

        hilicpos_msdata, hilicpos_misses = self.read_msdial_analyses(sample_db=self.community_db,
                                                                     map_df=community_map,
                                                                     chromatography='hilic',
                                                                     ionization='positive')

        community_db = self.community_db.copy(deep=True)
        community_db['mode'] = self.community_db['chromatography'] + self.community_db['ionization']

        joined_data = c18pos_msdata \
            .join(c18neg_msdata, how='outer') \
            .join(hilicpos_msdata, how='outer') \
            .join(community_db[['experiment', 'sample_type', 'colonization', 'sample_id', 'mouse_id', 'tissue_measurement']], how='inner')

        joined_data.index.name='run_id'

        # Get one combined sample data row for each set of runs
        # (c18pos, c18neg, hilicpos from the same sample)
        sample_data = joined_data \
            .groupby(['sample_id']) \
            .first() \
            .join(joined_data
                .join(community_db[['mode']])
                .reset_index()
                # Pivot so that we know which run ids are associated with each sample
                .pivot(index='sample_id', columns='mode', values='run_id')
            )

        metadata_columns = [
            'experiment',
            'sample_type',
            'colonization',
            'mouse_id',
            'tissue_measurement',
            'c18positive',
            'c18negative',
            'hilicpositive'
        ]

        metadata = sample_data[metadata_columns]
        sample_data = sample_data.drop(columns=metadata_columns)

        return (sample_data, metadata)


    def collapse_community_mouse_ids(self, istd_corrected_matrix, metadata):
        """
        Some community data (sample type = caecal) have multiple tissue measurements per mouse.

        We need to average those tissue measurements so that data from all sample types
        can be represented consistently.
        """

        metadata_cols = [
            'experiment',
            'sample_type',
            'colonization',
            'mouse_id',
            'tissue_measurement'
        ]

        # Extract community data with tissue measurements
        comm_with_tissue_measurements = istd_corrected_matrix.loc[
            metadata[
                (metadata['experiment'] == 'community') &
                (metadata['tissue_measurement'].notna())
            ].index.values
        ].join(metadata[metadata_cols])

        # Average the 3 tissue measurements performed for each mouse id
        comm_with_tissue_measurements_collapsed = comm_with_tissue_measurements \
            .groupby(['experiment', 'colonization', 'sample_type', 'mouse_id']) \
            .mean() \
            .reset_index()

        # Get the concatenated sample ids (e.g. "s0161,s0162,s0163")
        # for the tissue measurements performed for each mouse id
        comm_with_tissue_measurements_concatenated_sample_ids = comm_with_tissue_measurements \
            .reset_index() \
            .groupby(['experiment', 'colonization', 'sample_type', 'mouse_id'])['sample_id'] \
            .apply(','.join) \
            .reset_index()['sample_id']

        comm_with_tissue_measurements_collapsed = comm_with_tissue_measurements_collapsed \
            .join(comm_with_tissue_measurements_concatenated_sample_ids) \
            .set_index('sample_id') \
            .sort_index()

        metadata_cols.remove('tissue_measurement')

        comm_with_tissue_measurements_metadata = comm_with_tissue_measurements_collapsed[metadata_cols]

        comm_with_tissue_measurements_collapsed = comm_with_tissue_measurements_collapsed.drop(columns=metadata_cols)

        # Combine the averaged tissue measurements with the rest of the dataset
        new_matrix = pd.concat([
            istd_corrected_matrix.loc[metadata[metadata['tissue_measurement'].isna()].index.values],
            comm_with_tissue_measurements_collapsed
        ], sort=True).sort_index()

        new_metadata = pd.concat([
            metadata[metadata['tissue_measurement'].isna()],
            comm_with_tissue_measurements_metadata
        ], sort=False).sort_index()

        return (new_matrix, new_metadata)


    def normalize_by_istd(self, raw_ion_counts_matrix, metadata, dname_cpd_map, cpd_dname_map):
        """
        Normalize the raw ion counts within a mode and sample type based on
        internal standards compounds to account for differences between experiments
        """

        istd_corrected_matrix = raw_ion_counts_matrix.copy(deep=True)

        for mode in ['c18positive', 'c18negative', 'hilicpositive']:
            dnames_in_mode = list(filter(
                lambda dname: self.get_mode_from_dname(dname) == mode,
                istd_corrected_matrix.columns.values
            ))

            istd_corrected_matrix_in_mode = istd_corrected_matrix[dnames_in_mode]
            istd_corrected_matrix_other_modes = istd_corrected_matrix.drop(columns=dnames_in_mode)

            istds = self.ISTD_CHOICES[mode]
            istd_dnames = list(map(lambda istd: cpd_dname_map[f'{istd}.{mode}'], istds))
            istd_dnames = list(set(istd_corrected_matrix_in_mode.columns.values) & set(istd_dnames))

            sampletype_dfs = []
            all_istd_sums = []

            for sample_type, sample_ids in metadata.groupby(['sample_type']).groups.items():
                sample_type_data = istd_corrected_matrix_in_mode.loc[sample_ids]

                # Remove rows with all NaNs to account for mode-specific sample data that was removed
                istd_data = sample_type_data[istd_dnames].dropna(axis=0, how='all')

                # Drop any internal standards compound that has any nan
                # Ensures that each row is normalized consistently
                istd_data = istd_data.dropna(axis=1, how='any')

                removed_istd_dnames = list(set(istd_dnames) - set(istd_data.columns.values))

                #if removed_istd_dnames:
                    #print(f'{sample_type}, {mode}: ISTD columns removed with a NaN: {removed_istd_dnames}')
                    #print(f'{len(istd_data.columns)} of {len(istd_dnames)} columns remaining')

                # Sum the ISTD raw ion counts for each row
                istd_sums = istd_data.sum(axis=1)

                # Divide each raw ion count by the corresponding ISTD sum for the row
                sampletype_dfs.append(sample_type_data.divide(istd_sums, axis=0))

                all_istd_sums.append(istd_sums)

            all_istd_sums = pd.concat(all_istd_sums, sort=False)

            # Multiply corrected data by the median to bring it back to the original magnitude
            istd_corrected_matrix_in_mode = pd.concat(sampletype_dfs, sort=False) * all_istd_sums.median()

            istd_corrected_matrix = istd_corrected_matrix_in_mode \
                .join(istd_corrected_matrix_other_modes, how='outer')

        # Remove all internal standard columns
        istd_cols = list(filter(
            lambda dname: dname_cpd_map[dname].startswith('IS_'),
            istd_corrected_matrix.columns
        ))

        istd_corrected_matrix = istd_corrected_matrix.drop(columns=istd_cols)

        return istd_corrected_matrix.sort_index().sort_index(axis=1)


    def get_fold_change(self, istd_corrected_matrix, metadata):
        """
        Calculate the log2 of the ratio of the data to its germ-free mean
        from the same experiment and sample type
        """

        exp_sampletype_dfs = []

        istd_corrected_matrix = istd_corrected_matrix.add(1)

        for exp_sampletype, sample_ids in metadata.groupby(['experiment', 'sample_type']).groups.items():
            exp_sampletype_metadata = metadata.loc[sample_ids]
            germfree_metadata = exp_sampletype_metadata[exp_sampletype_metadata['colonization'] == 'germ-free']

            germfree_mean = istd_corrected_matrix.loc[germfree_metadata.index.values] \
                .mean(axis=0)

            cur_exp_sampletype_data = istd_corrected_matrix.loc[sample_ids] / germfree_mean

            exp_sampletype_dfs.append(np.log2(cur_exp_sampletype_data))

        return pd.concat(exp_sampletype_dfs, sort=True).sort_index()


    def collapse_replicates(self, fold_change_matrix, metadata):
        """
        Average the samples within the same experiment, sample type, and colonization
        """

        metadata_cols = ['experiment', 'sample_type', 'colonization']

        # Make sure we average based on the raw data instead of the log values
        matrix = 2 ** fold_change_matrix
        matrix = matrix.join(metadata[metadata_cols])

        return np.log2(matrix.groupby(metadata_cols).mean())


    def remove_dnames(self, matrix):
        dnames = set(self.METABOLITES_TO_REMOVE) & set(matrix.columns)
        return matrix.drop(columns=dnames)


    def collapse_modes(self, fold_change_matrix, mode_picker):
        """Collapses the 3 mode columns for each metabolite
        (.c18positive, .c18negative, .hilicpositive) into a single column
        by either picking a single mode, or averaging the values between multiple modes

        Parameters
        ----------
        fold_change_matrix : pd.DataFrame
            A fold change matrix with compound names instead of dnames as columns.

            Column names should be in the format <metabolite>.<mode>
        mode_picker : pd.DataFrame
            Mode picking definition file containing the preferred mode(s) for each metabolite.

            The preferred modes are specified in the "mode_pref" column.
        """

        def get_colname(cpd, feature):
            return cpd + '.' + features_to_suffixes[feature]

        mode_picker = mode_picker[mode_picker['mode_detected'].notnull()]

        mode_picker.index = mode_picker.index.map(lambda val: val.split('.')[0])

        features_to_suffixes = {
            'c18p': 'c18positive',
            'c18n': 'c18negative',
            'hilicp': 'hilicpositive',
        }

        for cpd, row in mode_picker.iterrows():
            #print("Processing compound {}:".format(cpd))

            features = list(map(lambda feature: feature.strip(), row['mode_detected'].split(',')))
            mode_prefs = list(map(lambda mode_pref: mode_pref.strip(), row['mode_pref'].split(',')))

            #print(f"Features: {features}; mode prefs: {mode_prefs}")

            feature_colnames = list(map(lambda feature: get_colname(cpd, feature), features))
            preferred_colnames = list(map(lambda pref: get_colname(cpd, pref), mode_prefs))
            remaining_colnames = [get_colname(cpd, feature) for feature in (set(features) - set(mode_prefs))]

            # If compounds have been removed from the fold change matrix as per
            # METABOLITES_TO_REMOVE, then skip over these missing columns here
            if (len(set(feature_colnames) & set(fold_change_matrix.columns)) < len(feature_colnames)):
                continue

            preferred_data = fold_change_matrix[preferred_colnames]

            # Average the data from the preferred modes
            fold_change_matrix[cpd] = preferred_data.apply(lambda values: np.log2(np.mean(2 ** values)), axis=1)

            if len(remaining_colnames) > 0:
                remaining_data = fold_change_matrix[remaining_colnames]

                # For any rows that have nans after extracting the data from the preferred modes,
                # average the data from the remaining modes so that we're not throwing away data.
                fold_change_matrix[cpd] = fold_change_matrix.apply(
                    lambda sample: np.log2(np.mean(2 ** remaining_data.loc[sample.name])) if pd.isna(sample[cpd]) else sample[cpd],
                    axis=1
                )

            fold_change_matrix = fold_change_matrix.drop(columns=feature_colnames)

        return fold_change_matrix.sort_index(axis=1)


    def run(self,
            collapse_mouse_ids=False,
            output_cpd_names=False,
            remove_dnames=False):
        # Dictionary of dnames to compound names
        # (dnames with multiple compounds concatenated together)
        dname_cpd_map = self.cpd_library \
            .groupby(['dname'])['Compound'] \
            .apply(lambda compounds: ', '.join(sorted(set(compounds)))) \
            .to_dict()

        dname_cpd_map = {dname: cpd.strip() + '.' + self.get_mode_from_dname(dname) for dname, cpd in dname_cpd_map.items()}

        cpd_dname_map = {value:key for key, value in dname_cpd_map.items()}

        non_community_matrix, non_community_metadata = self.get_non_community_matrix()
        community_matrix, community_metadata = self.get_community_matrix()

        raw_ion_counts_matrix = pd.concat([non_community_matrix, community_matrix], sort=True).sort_index()
        all_metadata = pd.concat([non_community_metadata, community_metadata], sort=False).sort_index()

        raw_ion_counts_matrix = self.sum_peaks(raw_ion_counts_matrix, dname_cpd_map)
        istd_corrected_matrix = self.normalize_by_istd(raw_ion_counts_matrix, all_metadata, dname_cpd_map, cpd_dname_map)

        if collapse_mouse_ids:
            istd_corrected_matrix, all_metadata = self.collapse_community_mouse_ids(istd_corrected_matrix, all_metadata)

        fold_change_matrix = self.get_fold_change(istd_corrected_matrix, all_metadata)

        result = {
            'raw_ion_counts_matrix': raw_ion_counts_matrix,
            'istd_corrected_matrix': istd_corrected_matrix,
            'fold_change_matrix': fold_change_matrix,
        }

        if collapse_mouse_ids:
            result['fold_change_replicates_collapsed_matrix'] = \
                self.collapse_replicates(fold_change_matrix, all_metadata)

        if remove_dnames:
            result = {key:self.remove_dnames(matrix) for (key,matrix) in result.items()}

        if output_cpd_names:
            result = {key:self.rename_matrix(matrix, dname_cpd_map) for (key,matrix) in result.items()}

        result['metadata'] = all_metadata

        return result
