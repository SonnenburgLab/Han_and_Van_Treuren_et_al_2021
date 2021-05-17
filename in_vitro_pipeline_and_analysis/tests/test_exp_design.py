import unittest
import pandas as pd
import numpy as np

from collections import defaultdict
from io import StringIO

from in_vitro_library_code.exp_design import (BiologicalSampleGroup, 
                                              Experiment,
                                              rebuild_bsg_without_samples,
                                              squeeze_metadata,
                                              get_bsg_with_samples,
                                              partition_samples)


class TestExpdesign_squeeze_metadata(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(None, squeeze_metadata(None))

        # all nan
        s = pd.Series([np.nan, np.nan, np.nan])
        self.assertTrue(np.isnan(squeeze_metadata(s)))

        # 1 non nan entry
        s = pd.Series([np.nan, 1.0, np.nan])
        self.assertEqual(1, squeeze_metadata(s))
        s = pd.Series([np.nan, '1.0', np.nan])
        self.assertEqual('1.0', squeeze_metadata(s))

        # good data, multiple entries
        s = pd.Series([3.0, 3.0, 3.0])
        self.assertEqual(3.0, squeeze_metadata(s))

        # good data, len 1
        s = pd.Series(['abc'])
        self.assertEqual('abc', squeeze_metadata(s))

        # inconsistent data
        s = pd.Series([3.0, 'abc', 3.0])
        self.assertEqual(None, squeeze_metadata(s))
        s = pd.Series([3.0, 4.0, 3.0])
        self.assertEqual(None, squeeze_metadata(s))


class TestExpdesign_BiologicalSampleGroup(unittest.TestCase):
    def setUp(self):
        cols = ['sample_type', 'clean_by_16s', 'chromatography', 'ionization',
                'run_designation', 'lcms_run_date', 'extraneous_column_1',
                'media', 'preculture_time', 'subculture_time', 'taxonomy',
                'culture_source', 'matrix_tube', 'extraneous_column_2',
                'od_sample_id']
        data = [['media_blank', '', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '', 'mm', '', '', '', '', '', np.nan, np.nan],
                ['media_blank', '', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '', 'mm', '', '', '', '', '', np.nan, np.nan],
                ['media_blank', '', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '', 'mm', '', '', '', '', '', np.nan, np.nan],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '12', 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 'od_005'],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '55', 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 np.nan],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 np.nan, 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 np.nan],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_X', 20190101,
                 '12', 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 'od_005'],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '55', 'mm', 12.0, 15.0, 'B. theta', 'c001', 'm005', np.nan,
                 np.nan],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 np.nan, 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 np.nan]]
        sids = ['s%s' % i for i in range(1, 10)]
        self.data = pd.DataFrame(data, index=sids, columns=cols)

    def test_simple(self):
        data = self.data.loc[['s1', 's2', 's3']]
        obs = BiologicalSampleGroup(data)

        # Skip __init__ and construct what we expect.
        exp_data = {'samples': data.index.values,
                    'sample_type': 'media_blank',
                    'clean_by_16s': '',
                    'chromatography': 'c18',
                    'ionization': 'positive',
                    'run_designation': 'c18_pos_1',
                    'lcms_run_date': 20190101,
                    'media': 'mm',
                    'preculture_time': '',
                    'subculture_time': '',
                    'taxonomy': '',
                    'culture_source': '',
                    'matrix_tube': '',
                    'removed_samples': set(),
                    'od_sample_ids': data['od_sample_id']}
        exp = BiologicalSampleGroup.__new__(BiologicalSampleGroup)
        for k,v in exp_data.items():
            exp.__dict__[k] = v

        self.assertEqual(obs, exp)

    def test_with_1_nan(self):
        data = self.data.loc[['s4', 's5', 's6']]
        obs = BiologicalSampleGroup(data)

        # Skip __init__ and construct what we expect.
        exp_data = {'samples': data.index.values,
                    'sample_type': 'supernatant',
                    'clean_by_16s': 'y',
                    'chromatography': 'c18',
                    'ionization': 'positive',
                    'run_designation': 'c18_pos_1',
                    'lcms_run_date': 20190101,
                    'media': 'mm',
                    'preculture_time': 12.0,
                    'subculture_time': 8.0,
                    'taxonomy': 'B. theta',
                    'culture_source': 'c001',
                    'matrix_tube': 'm005',
                    'removed_samples': set(),
                    'od_sample_ids': data['od_sample_id']}
        exp = BiologicalSampleGroup.__new__(BiologicalSampleGroup)
        for k,v in exp_data.items():
            exp.__dict__[k] = v

        self.assertEqual(obs, exp)

    def test_with_bad_grouping(self):
        data = self.data.loc[['s7', 's8', 's9']]
        self.assertRaises(ValueError, BiologicalSampleGroup, data)


class TestExpdesign_get_and_rebuild_bsg(unittest.TestCase):
    def setUp(self):
        cols = ['sample_type', 'clean_by_16s', 'chromatography', 'ionization',
                'run_designation', 'lcms_run_date', 'extraneous_column_1',
                'media', 'preculture_time', 'subculture_time', 'taxonomy',
                'culture_source', 'matrix_tube', 'extraneous_column_2',
                'od_sample_id']
        data = [['media_blank', '', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '', 'mm', '', '', '', '', '', np.nan, np.nan],
                ['media_blank', '', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '', 'mm', '', '', '', '', '', np.nan, np.nan],
                ['media_blank', '', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '', 'mm', '', '', '', '', '', np.nan, np.nan],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '12', 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 'od_005'],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '55', 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 np.nan],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 np.nan, 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 np.nan],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_X', 20190101,
                 '12', 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 'od_005'],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 '55', 'mm', 12.0, 15.0, 'B. theta', 'c001', 'm005', np.nan,
                 np.nan],
                ['supernatant', 'y', 'c18', 'positive', 'c18_pos_1', 20190101,
                 np.nan, 'mm', 12.0, 8.0, 'B. theta', 'c001', 'm005', np.nan,
                 np.nan]]
        sids = ['s%s' % i for i in range(1, 10)]
        self.data = pd.DataFrame(data, index=sids, columns=cols)

        bsgs = [frozenset(['s1', 's2', 's3']), frozenset(['s1']),
                frozenset(['s1', 's3']), frozenset(['s4', 's5', 's6']),
                frozenset(['s6'])]
        self.bsgs = {f:BiologicalSampleGroup(self.data.loc[f]) for f in bsgs}

    def test_get_bsg_with_samples(self):
        # Exact match
        obs_k, obs_bsg = get_bsg_with_samples(self.bsgs, ['s1', 's2', 's3'])
        exp_bsg = self.bsgs[frozenset(['s1', 's2', 's3'])]
        exp_k = frozenset(['s1', 's2', 's3'])
        self.assertEqual(obs_bsg, exp_bsg)
        self.assertEqual(obs_k, exp_k)
        # order doesn't matter
        obs_k, obs_bsg  = get_bsg_with_samples(self.bsgs, ['s2', 's1', 's3'])
        self.assertEqual(obs_bsg, exp_bsg)
        self.assertEqual(obs_k, exp_k)

        # More than 1, should fail.
        self.assertRaises(ValueError, get_bsg_with_samples, self.bsgs, ['s1'])

        # No match, should fail.
        self.assertRaises(ValueError, get_bsg_with_samples, self.bsgs,
                          ['s1', 's2', 'x'])

    def test_rebuild_bsg_without_samples(self):
        # Remove a single sample
        nsamples, nbsg = rebuild_bsg_without_samples(
                            self.bsgs[frozenset(['s1', 's2', 's3'])],
                            ['s2'])
        exp_nbsg = self.bsgs[frozenset(['s1', 's3'])]
        exp_nbsg.removed_samples = set(['s2'])
        exp_samples = frozenset(['s1', 's3'])

        self.assertEqual(nbsg, exp_nbsg)
        self.assertEqual(nsamples, exp_samples)

        # Remove 2 samples.
        nsamples, nbsg = rebuild_bsg_without_samples(
                            self.bsgs[frozenset(['s4', 's5', 's6'])],
                            ['s4', 's5'])
        exp_nbsg = self.bsgs[frozenset(['s6'])]
        exp_nbsg.removed_samples = set(['s5', 's4'])
        exp_samples = frozenset(['s6'])
        self.assertEqual(nbsg, exp_nbsg)
        self.assertEqual(nsamples, exp_samples)

        # Remove all samples..
        nsamples, nbsg = rebuild_bsg_without_samples(
                            self.bsgs[frozenset(['s4', 's5', 's6'])],
                            ['s4', 's5', 's6'])
        self.assertEqual(nsamples, None)
        self.assertEqual(nbsg, None)


class TestExpdesign_partition_samples(unittest.TestCase):
    def setUp(self):
        self.exp_str = ''.join(['sample_id\tmake_public\tcaution\tsample_type\tclean_by_16s\tculture_source\tmatrix_tube\ttaxonomy\texperiment\tod_sample_id\tpreculture_date\tpreculture_time\tsubculture_time\tmedia\tchromatography\tionization\tlcms_run_date\trun_designation\trun_order\n',
                                's1\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t1\n',
                                's2\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t2\n',
                                's3\ty\t\tqc\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t3\n',
                                's4\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t4\n',
                                's5\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t5\n',
                                's6\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\tod00001\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t6\n',
                                's7\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t7\n',
                                's8\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t8\n',
                                's9\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t9\n',
                                's10\ty\t\tsupernatant\ty\tc0009\t\tAnaerostipes sp. 3_2_56FAA BEI HM-220 904a\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t10\n',
                                's11\ty\t\tsupernatant\ty\tc0009\t\tAnaerostipes sp. 3_2_56FAA BEI HM-220 904a\t20190101\tod00002\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t11\n',
                                's12\ty\t\tsupernatant\ty\tc0009\t\tAnaerostipes sp. 3_2_56FAA BEI HM-220 904a\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t12\n',
                                's13\ty\t\tsupernatant\ty\tc0009\t\tAnaerostipes sp. 3_2_56FAA BEI HM-220 904a\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t13\n',
                                's14\ty\t\tsupernatant\ty\tc0010\t\tB. theta\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t14\n',
                                's15\ty\t\tmedia_blank\t\t\t\t\t20190101\t\t\t\t\tmm\tc18\tpositive\t20190105\tc18_pos_1\t15\n',
                                's16\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t16\n',
                                's17\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t17\n',
                                's18\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t18'])

    def test_simple(self):
        # Case 1: include the media blank in the bsgs
        md = pd.read_csv(StringIO(self.exp_str), sep='\t', index_col=0)
        bsg1 = BiologicalSampleGroup(md.loc[['s5', 's6', 's7', 's8', 's9']])
        bsg2 = BiologicalSampleGroup(md.loc[['s10', 's11', 's12', 's13']])
        bsg3 = BiologicalSampleGroup(md.loc[['s14']])
        bsg4 = BiologicalSampleGroup(md.loc[['s15']])
        bsgs = {frozenset(bsg.samples):bsg for bsg in (bsg1, bsg2, bsg3, bsg4)}
        samples = md.index.values

        obs = partition_samples(samples, bsgs, md)

        culture_sample_groups =  [frozenset(['s5', 's6', 's7', 's8', 's9']),
                                  frozenset(['s10', 's11', 's12', 's13']),
                                  frozenset(['s14'])]
        unspent_medias = {'mm': frozenset(['s15'])}
        media_in_samples = {frozenset(['s5', 's6', 's7', 's8', 's9']): 'mm',
                            frozenset(['s10', 's11', 's12', 's13']): 'mm',
                            frozenset(['s14']): 'mm'}
        cultured_in = defaultdict(list,
                                  {'mm': [frozenset(['s5', 's6', 's7', 's8', 's9']),
                                          frozenset(['s10', 's11', 's12', 's13']),
                                          frozenset(['s14'])]})
        ni_blank_samples = set(['s1', 's2', 's4', 's16', 's17', 's18'])
        qc_samples = set(['s3'])

        self.assertEqual(set(obs[0]), set(culture_sample_groups))
        self.assertEqual(obs[1], unspent_medias)
        self.assertEqual(obs[2], media_in_samples)
        # Check keys are the same then iterate through values since they are
        # lists.
        self.assertEqual(obs[3].keys(), cultured_in.keys())
        for k, v in obs[3].items():
            self.assertEqual(set(v), set(cultured_in[k]))
        self.assertEqual(obs[4], ni_blank_samples)
        self.assertEqual(obs[5], qc_samples)

        # Case 2: exclude the media blank from the bsgs, this should trigger an
        #         error because `exp_samples` includes this sample but it's not
        #         found in a bsg.
        bsgs = {frozenset(bsg.samples):bsg for bsg in (bsg1, bsg2, bsg3)}
        self.assertRaises(ValueError, partition_samples, exp_samples=samples,
                          bsgs=bsgs, md=md)

        # Case 3: remove the media blank from the `exp_samples`. No error.
        bsgs = {frozenset(bsg.samples):bsg for bsg in (bsg1, bsg2, bsg3, bsg4)}
        samples = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                   's11', 's12', 's13', 's14', 's16', 's17', 's18']
        obs = partition_samples(samples, bsgs, md)
        
        unspent_medias = {}
        self.assertEqual(set(obs[0]), set(culture_sample_groups))
        self.assertEqual(obs[1], unspent_medias)
        self.assertEqual(obs[2], media_in_samples)
        # Check keys are the same then iterate through values since they are
        # lists.
        self.assertEqual(obs[3].keys(), cultured_in.keys())
        for k, v in obs[3].items():
            self.assertEqual(set(v), set(cultured_in[k]))
        self.assertEqual(obs[4], ni_blank_samples)
        self.assertEqual(obs[5], qc_samples)

        # Case 4: no qc or blank samples passed. No error.
        samples = ['s5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13',
                   's14', 's15']
        obs = partition_samples(samples, bsgs, md)

        unspent_medias = {'mm': frozenset(['s15'])}
        ni_blank_samples = set()
        qc_samples = set()

        self.assertEqual(set(obs[0]), set(culture_sample_groups))
        self.assertEqual(obs[1], unspent_medias)
        self.assertEqual(obs[2], media_in_samples)
        # Check keys are the same then iterate through values since they are
        # lists.
        self.assertEqual(obs[3].keys(), cultured_in.keys())
        for k, v in obs[3].items():
            self.assertEqual(set(v), set(cultured_in[k]))
        self.assertEqual(obs[4], ni_blank_samples)
        self.assertEqual(obs[5], qc_samples)


class TestExpdesign_Experiment(unittest.TestCase):
    def setUp(self):
        self.exp_str = ''.join(['sample_id\tmake_public\tcaution\tsample_type\tclean_by_16s\tculture_source\tmatrix_tube\ttaxonomy\texperiment\tod_sample_id\tpreculture_date\tpreculture_time\tsubculture_time\tmedia\tchromatography\tionization\tlcms_run_date\trun_designation\trun_order\n',
                                's1\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t1\n',
                                's2\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t2\n',
                                's3\ty\t\tqc\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t3\n',
                                's4\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t4\n',
                                's5\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t5\n',
                                's6\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\tod00001\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t6\n',
                                's7\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t7\n',
                                's8\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t8\n',
                                's9\ty\t\tsupernatant\ty\tc0002\t\tAcidaminococcus intestini D21 BEI HM-81\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t9\n',
                                's10\ty\t\tsupernatant\ty\tc0009\t\tAnaerostipes sp. 3_2_56FAA BEI HM-220 904a\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t10\n',
                                's11\ty\t\tsupernatant\ty\tc0009\t\tAnaerostipes sp. 3_2_56FAA BEI HM-220 904a\t20190101\tod00002\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t11\n',
                                's12\ty\t\tsupernatant\ty\tc0009\t\tAnaerostipes sp. 3_2_56FAA BEI HM-220 904a\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t12\n',
                                's13\ty\t\tsupernatant\ty\tc0009\t\tAnaerostipes sp. 3_2_56FAA BEI HM-220 904a\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t13\n',
                                's14\ty\t\tsupernatant\ty\tc0010\t\tB. theta\t20190101\t\t20190101\t24.0\t12.0\tmm\tc18\tpositive\t20190105\tc18_pos_1\t14\n',
                                's15\ty\t\tmedia_blank\t\t\t\t\t20190101\t\t\t\t\tmm\tc18\tpositive\t20190105\tc18_pos_1\t15\n',
                                's16\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t16\n',
                                's17\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t17\n',
                                's18\ty\t\tni_blank\t\t\t\t\t20190101\t\t\t\t\t\tc18\tpositive\t20190105\tc18_pos_1\t18'])

        self.md = pd.read_csv(StringIO(self.exp_str), sep='\t', index_col=0)


    def test_simple(self):

        bsg1 = BiologicalSampleGroup(self.md.loc[['s5', 's6', 's7', 's8', 's9']])
        bsg2 = BiologicalSampleGroup(self.md.loc[['s10', 's11', 's12', 's13']])
        bsg3 = BiologicalSampleGroup(self.md.loc[['s14']])
        bsg4 = BiologicalSampleGroup(self.md.loc[['s15']])
        self.bsgs = {frozenset(bsg.samples):bsg for bsg in (bsg1, bsg2, bsg3, bsg4)}
        samples = self.md.index.values

        tmp = partition_samples(samples, self.bsgs, self.md)
        obs = Experiment(culture_sample_groups=tmp[0],
                         unspent_medias=tmp[1],
                         media_in_samples=tmp[2],
                         cultured_in=tmp[3],
                         ni_blank_samples=tmp[4],
                         qc_samples=tmp[5],
                         md=self.md)

        exp_data = {'chromatography': 'c18',
                    'culture_date': 20190101,
                    'culture_sample_groups': [frozenset(['s5', 's6', 's7', 's8', 's9']),
                                              frozenset(['s10', 's11', 's12', 's13']),
                                              frozenset(['s14'])],
                    'cultured_in': defaultdict(list,
                                  {'mm': [frozenset(['s5', 's6', 's7', 's8', 's9']),
                                          frozenset(['s10', 's11', 's12', 's13']),
                                          frozenset(['s14'])]}),
                    'exp_id': 20190101,
                    'has_needed_unspent_media': True,
                    'ionization': 'positive',
                    'lcms_run_date': 20190105,
                    'media_in_samples': {frozenset(['s5', 's6', 's7', 's8', 's9']): 'mm',
                                         frozenset(['s10', 's11', 's12', 's13']): 'mm',
                                         frozenset(['s14']): 'mm'},
                    'ni_blanks': set(['s1', 's2', 's4', 's16', 's17', 's18']),
                    'other_samples': None,
                    'qc_samples': set(['s3']),
                    'removed_samples': set(),
                    'samples': {'all': set(['s1', 's2', 's3', 's4', 's5',
                                            's6', 's7', 's8', 's9', 's10',
                                            's11', 's12', 's13', 's14',
                                            's15','s16', 's17', 's18']),
                                'qcs': set(['s3']),
                                'ni_blanks': set(['s1', 's2', 's4', 's16',
                                                  's17', 's18']),
                                'unspent_medias': set(['s15']),
                                'supernatants': set(['s5', 's6', 's7', 's8',
                                                     's9', 's10', 's11', 's12',
                                                     's13', 's14'])},
                    'unspent_medias': {'mm':set(['s15'])}}
        
        exp = Experiment.__new__(Experiment)
        for k,v in exp_data.items():
            exp.__dict__[k] = v

        self.assertEqual(obs, exp)

if __name__ == '__main__':
    unittest.main()