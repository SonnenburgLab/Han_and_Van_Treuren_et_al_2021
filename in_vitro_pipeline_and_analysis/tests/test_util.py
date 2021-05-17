import unittest
import pandas as pd
import numpy as np

from in_vitro_library_code.util import (fc_and_delta, media_based_transform,
                                        superset, intra_replicate_correlation,
                                        istd_tic_transform, join_across_modes,
                                        add_multipeaks,
                                        combine_features_in_fc_data)

class TestUtil_fc_and_delta(unittest.TestCase):
    def test_simple_example(self):
        sn_samples = ['s1', 's2', 's3']
        mb_samples = ['s4', 's5', 's6']
        fnames = ['f%s' % i for i in range(10)]
        _test_msdata = np.arange(60).reshape(6, 10)
        test_msdata = pd.DataFrame(_test_msdata, index=sn_samples + mb_samples,
                                   columns=fnames)

        exp = np.array([[-2.71080821,  4.90135525,  5.32805789],
                        [-2.34945434,  4.90135525,  5.36425238],
                        [-2.12987172,  4.90135525,  5.39952644],
                        [-1.96982151,  4.90135525,  5.43392732],
                        [-1.84327365,  4.90135525,  5.46749861],
                        [-1.73847989,  4.90135525,  5.50028063],
                        [-1.64907566,  4.90135525,  5.53231074],
                        [-1.57119703,  4.90135525,  5.56362367],
                        [-1.5023072 ,  4.90135525,  5.5942517 ],
                        [-1.4406427 ,  4.90135525,  5.62422496]])

        exp_summary_df = pd.DataFrame(exp, index=fnames,
                                      columns=['b/m', 'b-m', 'media_mean'])

        tmp = np.array([[-5.32805789, -4.36425238, -3.81456394, -3.43392732, -3.14557051,
                -2.91531813, -2.72495582, -2.56362367, -2.4243267 , -2.30229687],
               [-1.86862627, -1.77928988, -1.69908672, -1.6265724 , -1.56060801,
                -1.50028063, -1.4448479 , -1.39369867, -1.34632419, -1.30229687],
               [-0.93574046, -0.90482076, -0.87596449, -0.84896482, -0.82364242,
                -0.79984091, -0.77742324, -0.75626875, -0.73627071, -0.71733436]])
        exp_fc_df = pd.DataFrame(tmp.T, index=fnames, columns=sn_samples)

        tmp = np.array([[ 5.357552  ,  5.357552  ,  5.357552  ,  5.357552  ,  5.357552,
                 5.357552  ,  5.357552  ,  5.357552  ,  5.357552  ,  5.357552  ],
               [ 4.95419631,  4.95419631,  4.95419631,  4.95419631,  4.95419631,
                 4.95419631,  4.95419631,  4.95419631,  4.95419631,  4.95419631],
               [ 4.39231742,  4.39231742,  4.39231742,  4.39231742,  4.39231742,
                 4.39231742,  4.39231742,  4.39231742,  4.39231742,  4.39231742]])
        exp_delta_df = pd.DataFrame(tmp.T, index=fnames, columns=sn_samples)

        obs_summary_df, obs_fc_df, obs_delta_df = \
            fc_and_delta(sn_samples, mb_samples, test_msdata)
        pd.testing.assert_frame_equal(obs_summary_df, exp_summary_df)
        pd.testing.assert_frame_equal(obs_fc_df, exp_fc_df)
        pd.testing.assert_frame_equal(obs_delta_df, exp_delta_df)


class TestUtil_superset(unittest.TestCase):
    def setUp(self):
        self.data_in_1 = [frozenset([1,2,3])]
        self.data_in_2 = [frozenset([1,2,3]), frozenset([3,4,5])]
        self.data_in_3 = [frozenset([1,2,3]), frozenset([1,2,3])]
        self.data_in_4 = [('a', 'b', 'c'), ('c', 'd', 'e')]
        self.starting_set_1 = set([1,2,3])
        self.starting_set_2 = set([7,8,9])

    def test_errors(self):
        self.assertRaises(ValueError, superset, 'ab')
        self.assertRaises(ValueError, superset, [1,2,3], starting_set='144qz')
        self.assertRaises(ValueError, superset, '14', starting_set='144qz')

    def test_no_starting_set(self):
        self.assertEqual(set([1,2,3]), superset(self.data_in_1))
        self.assertEqual(set([1,2,3,4,5]), superset(self.data_in_2))
        self.assertEqual(set([1,2,3]), superset(self.data_in_3))

    def test_with_starting_set(self):
        self.assertEqual(set([1,2,3]),
                         superset(self.data_in_1,
                                  starting_set=self.starting_set_1))
        self.assertEqual(set([1,2,3,7,8,9]),
                         superset(self.data_in_1,
                                  starting_set=self.starting_set_2))
        self.assertEqual(set([1,2,3,7,8,9,4,5]),
                         superset(self.data_in_2,
                                  starting_set=self.starting_set_2))


class TestUtil_intra_replicate_correlation(unittest.TestCase):
    def setUp(self):
        data = np.array([[0, 0, 5000, 10000, 0, 10000, 0, 0, 60000, 7000],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1000, 0, 10000, 100, 1000, 50000, 0, 0, 0, 2000],
                         [5000, 0, 1000, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
        data_w_1nan = data.copy()
        data_w_1nan[0, 0] = np.nan
        data_w_2nan = data_w_1nan.copy()
        data_w_2nan[3, 4] = np.nan

        self.samples = ['s%i' % i for i in range(4)]
        self.features = ['f%i' % i for i in range(10)]

        self.data_1 = pd.DataFrame(data, index=self.samples,
                                   columns=self.features)
        self.data_2 = pd.DataFrame(data_w_1nan, index=self.samples,
                                   columns=self.features)
        self.data_3 = pd.DataFrame(data_w_2nan, index=self.samples,
                                   columns=self.features)

    def test_no_nan_in_input(self):
        obs = intra_replicate_correlation(self.data_1, ['s3', 's0', 's2'])
        exp = np.array([[0., -0.19276601, -0.1057971],
                        [-0.19276601, 0., -0.00958891],
                        [-0.1057971, -0.00958891, 0.]])
        np.testing.assert_allclose(obs, exp, atol=1e-7)

        obs = intra_replicate_correlation(self.data_1, ['s3', 's0', 's2'],
                                          as_squareform=False)
        exp = np.array([-0.19276601, -0.1057971, -0.00958891])
        np.testing.assert_allclose(obs, exp, atol=1e-7)

        obs = intra_replicate_correlation(self.data_1, ['s3'])
        exp = np.array([[0., 1],
                        [1., 0]])
        np.testing.assert_allclose(obs, exp)

        obs = intra_replicate_correlation(self.data_1, ['s0', 's1', 's3'])
        exp = np.array([[0., np.nan, -0.19276601],
                        [np.nan, 0., np.nan],
                        [-0.19276601, np.nan, 0.]])
        np.testing.assert_allclose(obs[~np.isnan(obs)], exp[~np.isnan(exp)])
        np.testing.assert_equal(np.isnan(obs), np.isnan(exp))

        obs = intra_replicate_correlation(self.data_1, ['s0', 's1', 's3'],
                                          as_squareform=False)
        exp = np.array([np.nan, -0.19276601, np.nan])
        np.testing.assert_allclose(obs[~np.isnan(obs)], exp[~np.isnan(exp)])
        np.testing.assert_equal(np.isnan(obs), np.isnan(exp))

    def test_nans_in_input(self):
        obs = intra_replicate_correlation(self.data_3, ['s0', 's3', 's2'])
        exp = np.array([[0., -0.13094623, -0.06101526],
                        [ -0.13094623, 0., 0.05192668],
                        [-0.06101526, 0.05192668, 0.]])
        np.testing.assert_allclose(obs, exp, atol=1e7)

        obs = intra_replicate_correlation(self.data_3, ['s0', 's3', 's2'],
                                          as_squareform=False)
        exp = np.array([-0.13094623, -0.06101526, 0.05192668])
        np.testing.assert_allclose(obs, exp)

        obs = intra_replicate_correlation(self.data_2, ['s3', 's2'],
                                          as_squareform=False)
        exp = np.array([-0.10579715])
        np.testing.assert_allclose(obs, exp)



class TestUtil_media_based_transform(unittest.TestCase):
    def setUp(self):
        snames = ['s%s' % i for i in range(6)]
        fnames = ['f%s' % i for i in range(10)]
        tmp = np.arange(60).reshape(6, 10)
        self.msdata_1 = pd.DataFrame(tmp, index=snames, columns=fnames)

        fnames = ['f%s' % i for i in range(6)]
        snames = ['s%s' % i for i in range(8)]
        tmp = np.array([[100, 10, 0, 50, 70, 100.],
                        [30, 1, 10, 10, 90, 200],
                        [1, 1, 1000, 10, 70, 30],
                        [40, 80, 100, 10, 50, 100],
                        [300, 200, 100, 10, 1, 1],
                        [900, 1, 1, 1, 1, 100],
                        [100, 200, 800, 1, 1, 100],
                        [0, 100, 1, 100, 1, 1000]])
        self.msdata_2 = pd.DataFrame(tmp, index=snames, columns=fnames)


    def test_error_conditions(self):
        # Not perfect overlap between msdata and sample map samples.
        self.assertRaises(ValueError, media_based_transform, 
                          data=self.msdata_1, sample_map={'s1':'s4'})

        sample_map = {frozenset(['s3', 's4', 's5']): frozenset(['s0', 's1',
                                                                's2']),
                      frozenset(['s9'])            : frozenset(['s0', 's1'])}
        self.assertRaises(ValueError, media_based_transform, 
                          data=self.msdata_1, sample_map=sample_map)

        # log transformed data with delta.
        sample_map = {frozenset(['s0', 's1', 's2']): frozenset(['s3', 's4',
                                                                's5'])}
        self.assertRaises(ValueError, media_based_transform, data=self.msdata_1,
                          sample_map=sample_map, fc_or_delta='delta',
                          data_is_log_transformed=True)

        # An important error, if we've assigned a sample to be transformed by
        # more than one sample we'll have unpredictable results.
        sample_map = {frozenset(['s3', 's5']): frozenset(['s0', 's1']),
                      frozenset(['s3', 's4']): frozenset(['s2'])}
        self.assertRaises(ValueError, media_based_transform, data=self.msdata_1,
                          sample_map=sample_map, fc_or_delta='delta',
                          data_is_log_transformed=False)
        # If values of the sample map overlap, that's fine. We can use the same
        # samples to normalize different samples, but we can't normalize the
        # same sample twice.
        sample_map = {frozenset(['s3', 's5']): frozenset(['s0', 's1']),
                      frozenset(['s3', 's5']): frozenset(['s0', 's1'])}
        self.assertRaises(ValueError, media_based_transform, data=self.msdata_1,
                          sample_map=sample_map, fc_or_delta='delta',
                          data_is_log_transformed=False)

    def test_simple(self):
        # Calculations for log transformed fc
        # tmp = self.msdata_1.values
        # tmp[:3, :] - tmp[3:, :].mean(0)
        # tmp[3:, :] - tmp[3:, :].mean(0)
        exp_fc = np.array([[-40., -40., -40., -40., -40., -40., -40., -40., -40., -40.],
                           [-30., -30., -30., -30., -30., -30., -30., -30., -30., -30.],
                           [-20., -20., -20., -20., -20., -20., -20., -20., -20., -20.],
                           [-10., -10., -10., -10., -10., -10., -10., -10., -10., -10.],
                           [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
                           [ 10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.,  10.]])
        exp_fc = pd.DataFrame(exp_fc, index=self.msdata_1.index,
                              columns=self.msdata_1.columns)

        sample_map = {frozenset(['s0', 's1', 's2']): frozenset(['s3', 's4',
                                                                's5']),
                      frozenset(['s3', 's4', 's5']): frozenset(['s3', 's4',
                                                                's5'])}

        obs_fc = media_based_transform(self.msdata_1, sample_map,
                                       fc_or_delta='fc',
                                       data_is_log_transformed=True)
        pd.testing.assert_frame_equal(obs_fc, exp_fc)

        # Same as logged 'fc' data.
        exp_delta = exp_fc
        obs_delta = media_based_transform(self.msdata_1, sample_map,
                                          fc_or_delta='delta',
                                          data_is_log_transformed=False)
        pd.testing.assert_frame_equal(obs_delta, exp_delta)
                                      

        # Do 'fc' assuming data is not logged.
        # tmp = self.msdata_1.values
        # tmp[:3, :] / tmp[3:, :].mean(0)
        # tmp[3:, :] / tmp[3:, :].mean(0)
        exp_fc = np.array([[ 0.        ,  0.02439024,  0.04761905,  0.06976744,  0.09090909,
                             0.11111111,  0.13043478,  0.14893617,  0.16666667,  0.18367347],
                           [ 0.25      ,  0.26829268,  0.28571429,  0.30232558,  0.31818182,
                             0.33333333,  0.34782609,  0.36170213,  0.375     ,  0.3877551 ],
                           [ 0.5       ,  0.51219512,  0.52380952,  0.53488372,  0.54545455,
                             0.55555556,  0.56521739,  0.57446809,  0.58333333,  0.59183673],
                           [ 0.75      ,  0.75609756,  0.76190476,  0.76744186,  0.77272727,
                             0.77777778,  0.7826087 ,  0.78723404,  0.79166667,  0.79591837],
                           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
                             1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
                           [ 1.25      ,  1.24390244,  1.23809524,  1.23255814,  1.22727273,
                             1.22222222,  1.2173913 ,  1.21276596,  1.20833333,  1.20408163]])
        exp_fc = pd.DataFrame(exp_fc, index=self.msdata_1.index,
                              columns=self.msdata_1.columns)

        sample_map = {frozenset(['s0', 's1', 's2']): frozenset(['s3', 's4',
                                                                's5']),
                      frozenset(['s3', 's4', 's5']): frozenset(['s3', 's4',
                                                                's5'])}
        obs_fc = media_based_transform(self.msdata_1, sample_map,
                                       fc_or_delta='fc',
                                       data_is_log_transformed=False)
        pd.testing.assert_frame_equal(obs_fc, exp_fc)

    def test_ordering_and_single_sample(self):
        # Test `fc` calculates okay when sample orders switch.
        sample_map_1 = {frozenset(['s0', 's1', 's7']): frozenset(['s5', 's2']),
                        frozenset(['s6', 's4']):       frozenset(['s3']),
                        frozenset(['s5', 's2']): frozenset(['s5', 's2']),
                        frozenset(['s3']):       frozenset(['s3'])}

        # fc, log_transformed=False
        exp_fc = np.array([[100/450.5, 10/1, 0/500.5, 50/5.5, 70/35.5, 100/65],
                           [30/450.5, 1/1, 10/500.5, 10/5.5, 90/35.5, 200/65],
                           [1/450.5, 1, 1000/500.5, 10/5.5, 70/35.5, 30/65],
                           [1, 1, 1, 1, 1, 1],
                           [300/40, 200/80, 100/100, 10/10, 1/50, 1/100],
                           [900/450.5, 1, 1/500.5, 1/5.5, 1/35.5, 100/65],
                           [100/40, 200/80, 800/100, 1/10, 1/50, 100/100],
                           [0, 100, 1/500.5, 100/5.5, 1/35.5, 1000/65]])
        exp_fc = pd.DataFrame(exp_fc, index=self.msdata_2.index,
                              columns=self.msdata_2.columns)
        obs_fc = media_based_transform(self.msdata_2, sample_map_1,
                                       fc_or_delta='fc',
                                       data_is_log_transformed=False)
        pd.testing.assert_frame_equal(obs_fc, exp_fc)

        # Change sample_map order and show it doesn't matter
        sample_map_1 = {frozenset(['s7', 's0', 's1']): frozenset(['s5', 's2']),
                        frozenset(['s4', 's6']):       frozenset(['s3']),
                        frozenset(['s2', 's5']): frozenset(['s5', 's2']),
                        frozenset(['s3']):       frozenset(['s3'])}

        obs_fc = media_based_transform(self.msdata_2, sample_map_1,
                                       fc_or_delta='fc',
                                       data_is_log_transformed=False)
        pd.testing.assert_frame_equal(obs_fc, exp_fc)

        # fc, log_transformed = True
        exp_fc = np.array([[100 - 450.5, 10 - 1, 0 - 500.5, 50 - 5.5, 70 - 35.5, 100 - 65],
                           [30 - 450.5, 1 - 1, 10 - 500.5, 10 - 5.5, 90 - 35.5, 200 - 65],
                           [1 - 450.5, 1 - 1, 1000 - 500.5, 10 - 5.5, 70 - 35.5, 30 - 65],
                           [0, 0, 0, 0, 0, 0],
                           [300 - 40, 200 - 80, 100 - 100, 10 - 10, 1 - 50, 1 - 100],
                           [900 - 450.5, 1 - 1, 1 - 500.5, 1 - 5.5, 1 - 35.5, 100 - 65],
                           [100 - 40, 200 - 80, 800 - 100, 1 - 10, 1 - 50, 100 - 100],
                           [0 - 450.5, 100 - 1, 1 - 500.5, 100 - 5.5, 1 - 35.5, 1000 - 65]])
        exp_fc = pd.DataFrame(exp_fc, index=self.msdata_2.index,
                              columns=self.msdata_2.columns)
        obs_fc = media_based_transform(self.msdata_2, sample_map_1,
                                       fc_or_delta='fc',
                                       data_is_log_transformed=True)
        pd.testing.assert_frame_equal(obs_fc, exp_fc)


class TestUtil_istd_tic_transform(unittest.TestCase):
    def setUp(self):
        fnames = ['f%s' % i for i in range(10)]
        snames = ['s%s' % i for i in range(5)]
        np.random.seed(2362463)
        _data = np.random.uniform(low=100, high=10000, size=50).reshape(5, 10)
        self.data = pd.DataFrame(_data, index=snames, columns=fnames)

        _data = self.data.copy()
        _data.iloc[[1,3,4], [3,4,5]] = np.nan
        self.data_w_nans = _data

        # example by hand
        fnames = ['f%s' % i for i in range(5)]
        snames = ['s%s' % i for i in range(3)]
        _data = np.array([[10,  20, 30, 40,  50],
                          [0,    0,  0,  0, 100],
                          [100, 10, 50,  0,  60]])
        self.data_by_hand = pd.DataFrame(_data, index=snames, columns=fnames)

    def test_errors(self):
        istds = ['f3']
        self.assertRaises(ValueError, istd_tic_transform, data=self.data,
                          istds=istds, by='nonsense_method')

    def test_median(self):
        # Test simple by hand.
        istds = ['f0', 'f4']
        obs = istd_tic_transform(self.data_by_hand, istds, by='median')
        exp_data = np.array([[10 * 100/60.,  20 * 100/60., 30 * 100/60., 40 * 100/60.,  50 * 100/60.],
                             [0,    0,  0,  0, 100],
                             [100 * 100/160., 10 * 100/160., 50 * 100/160.,  0 * 100/160.,  60 * 100/160.]])
        exp = pd.DataFrame(exp_data, index=self.data_by_hand.index,
                           columns=self.data_by_hand.columns)
        pd.testing.assert_frame_equal(obs, exp)


        # Test procedurally generated.
        istds = ['f0', 'f1', 'f7']
        obs = istd_tic_transform(self.data, istds, by='median')

        exp_data = np.array([[  4130.3524949 ,   4923.1915015 ,    197.17369805,    614.76632205,
                             3212.5529005 ,    773.15358468,   5675.48733387,   3910.12500012,
                             4104.11911036,   4646.79792592],
                          [  1941.5004799 ,   8943.03283787,   1317.9204872 ,   9207.74179653,
                             8015.58300824,   8948.49148149,   2297.84606839,   2079.13567874,
                             3130.99972895,    287.08382712],
                          [   403.83014577,   4314.27028137,   7043.60534803,   9912.32805089,
                             3933.14175991,   3764.81254616,  18316.94866739,   8245.56856938,
                            16218.82155242,  14585.67996049],
                          [  4864.01380631,   4597.76574516,   1471.4089366 ,    808.6540427 ,
                             7138.96709619,   1556.21819815,   3225.60564937,   3501.88944504,
                             7134.72827824,   1883.44070918],
                          [   212.2031071 ,   9957.29257227,   4442.48803919,   6222.06661503,
                             9639.08038803,   5700.31232747,   6013.01354265,   2794.17331715,
                             9688.17568922,   6895.57878466]])

        exp = pd.DataFrame(exp_data, index=self.data.index,
                           columns=self.data.columns)
        pd.testing.assert_frame_equal(obs, exp)

        # Switch istd order, show it doesn't matter.
        istds = ['f1', 'f7', 'f0']
        obs = istd_tic_transform(self.data, istds, by='median')
        pd.testing.assert_frame_equal(obs, exp)

        # Test when nans in fd, but not part of istds.
        istds = ['f0', 'f6']
        obs = istd_tic_transform(self.data, istds, by='median')

        exp_data = np.array([[  3981.58921394,   4745.87246603,    190.07207506,    592.62422759,
                             3096.84609089,    745.30684179,   5471.07279107,   3769.29367284,
                             3956.3006796 ,   4479.43378296],
                          [  4329.05109549,  19940.68325229,   2938.62669   ,  20530.91674404,
                            17872.70658036,  19952.85463591,   5123.61090952,   4635.94249959,
                             6981.33116471,    640.12374406],
                          [   203.90550593,   2178.39969026,   3556.51980701,   5005.02076773,
                             1985.95689022,   1900.96260771,   9248.75649908,   4163.42575827,
                             8189.35151068,   7364.73114478],
                          [  5683.56506145,   5372.45612174,   1719.33073305,    944.90641822,
                             8341.83157747,   1818.42974367,   3769.09694356,   4091.93256669,
                             8336.87854926,   2200.78688841],
                          [   322.219187  ,  15119.62176843,   6745.68296309,   9447.87884402,
                            14636.43340845,   8655.62257287,   9130.44281801,   4242.80429686,
                            14710.98202487,  10470.5714271 ]])

        exp = pd.DataFrame(exp_data, index=self.data.index,
                           columns=self.data.columns)
        pd.testing.assert_frame_equal(obs, exp)

        # Switch istd order, show it doesn't matter.
        istds = ['f6', 'f0']
        obs = istd_tic_transform(self.data, istds, by='median')
        pd.testing.assert_frame_equal(obs, exp)

    def test_mean(self):
        # Test simple by hand.
        istds = ['f0', 'f4']
        obs = istd_tic_transform(self.data_by_hand, istds, by='mean')
        c = 106.66666666666667
        exp_data = np.array([[10 * c / 60.,  20 * c / 60., 30 * c / 60., 40 * c / 60.,  50 * c / 60.],
                             [0,    0,  0,  0, 100 * c / 100],
                             [100 * c / 160., 10 * c / 160., 50 * c / 160.,  0 * c / 160.,  60 * c / 160.]])
        exp = pd.DataFrame(exp_data, index=self.data_by_hand.index,
                           columns=self.data_by_hand.columns)
        pd.testing.assert_frame_equal(obs, exp)


        # Test procedurally generated.
        istds = ['f0', 'f1', 'f7']
        obs = istd_tic_transform(self.data, istds, by='mean')

        exp_data = np.array([[  4368.25058482,   5206.75515762,    208.53041548,    650.17534197,
                              3397.58800338,    817.68531925,   6002.38136962,   4135.33853093,
                              4340.50622221,   4914.44199558],
                           [  2053.32610648,   9458.12941446,   1393.82944823,   9738.08495456,
                              8477.2607681 ,   9463.90246246,   2430.19631968,   2198.88875243,
                              3311.33757082,    303.61914564],
                           [   427.08976355,   4562.76158099,   7449.29959823,  10483.25363482,
                              4159.68099917,   3981.65643896,  19371.95961537,   8720.49292884,
                             17152.98556696,  15425.77905786],
                           [  5144.16897353,   4862.58568245,   1556.15845276,    855.23051587,
                              7550.15394726,   1645.85251806,   3411.3925583 ,   3703.58961739,
                              7545.67098388,   1991.92223654],
                           [   224.42548132,  10530.80800144,   4698.36436459,   6580.44226576,
                             10194.26758228,   6028.63622145,   6359.34825336,   2955.11079062,
                             10246.19065141,   7292.74707085]])

        exp = pd.DataFrame(exp_data, index=self.data.index,
                           columns=self.data.columns)
        pd.testing.assert_frame_equal(obs, exp)

        # Switch istd order, show it doesn't matter.
        istds = ['f1', 'f7', 'f0']
        obs = istd_tic_transform(self.data, istds, by='mean')
        pd.testing.assert_frame_equal(obs, exp)

        # Test when nans in fd, but not part of istds.
        istds = ['f0', 'f6']
        obs = istd_tic_transform(self.data, istds, by='mean')

        exp_data = np.array([[  3839.36639332,   4576.34936052,    183.28267887,    571.45562262,
                              2986.2263955 ,    718.68439644,   5275.64544729,   3634.65407316,
                              3814.98116831,   4319.4278976 ],
                           [  4174.41689686,  19228.39977334,   2833.65860964,  19797.55005748,
                             17234.29146387,  19240.13639375,   4940.59494375,   4470.34610503,
                              6731.95721965,    617.25845095],
                           [   196.62197802,   2100.58700499,   3429.48051407,   4826.24085535,
                              1915.0182837 ,   1833.06000666,   8918.3898626 ,   4014.70771556,
                              7896.82694116,   7101.66210873],
                           [  5480.54746955,   5180.55137662,   1657.91604327,    911.15425324,
                              8043.86040258,   1753.47522593,   3634.46437107,   3945.76826894,
                              8039.0842971 ,   2122.17452988],
                           [   310.70948091,  14579.54715527,   6504.72639866,   9110.39953468,
                             14113.61833859,   8346.44274785,   8804.30235971,   4091.25084371,
                             14185.50406993,  10096.56142211]])

        exp = pd.DataFrame(exp_data, index=self.data.index,
                           columns=self.data.columns)
        pd.testing.assert_frame_equal(obs, exp)

        # Switch istd order, show it doesn't matter.
        istds = ['f6', 'f0']
        obs = istd_tic_transform(self.data, istds, by='mean')
        pd.testing.assert_frame_equal(obs, exp)

    def test_where_nans_infs_would_be_introduced(self):
        # Test that when istd tic is 0, we don't introduce nans or infs.
        istds = ['f0', 'f1']
        obs = istd_tic_transform(self.data_by_hand, istds, by='mean')
        
        c = 140/3.
        exp_data = np.array([[10 * (c/30),  20 * (c/30), 30 * (c/30), 40 * (c/30),  50 * (c/30)],
                             [0,    0,  0,  0, 100],
                             [100 * (c/110), 10 * (c/110), 50 * (c/110),  0,  60 * (c/110)]])
        exp = pd.DataFrame(exp_data, index=self.data_by_hand.index,
                           columns=self.data_by_hand.columns)
        pd.testing.assert_frame_equal(obs, exp)


class TestUtil_join_across_modes(unittest.TestCase):
    def setUp(self):
        c18p_snames = ['c18p_s%s' % i for i in range(3)]
        c18p_fnames = ['c18p_f%s' % i for i in range(10)]
        c18p_data = np.arange(30).reshape(3, 10)
        self.simple_c18p_data = pd.DataFrame(c18p_data, index=c18p_snames,
                                             columns=c18p_fnames)

        c18n_snames = ['c18n_s%s' % i for i in range(3)]
        c18n_fnames = ['c18n_f%s' % i for i in range(10)]
        c18n_data = np.arange(30).reshape(3, 10) + 100
        self.simple_c18n_data = pd.DataFrame(c18n_data, index=c18n_snames,
                                             columns=c18n_fnames)

        hilicp_snames = ['hilicp_s%s' % i for i in range(3)]
        hilicp_fnames = ['hilicp_f%s' % i for i in range(10)]
        hilicp_data = np.arange(30).reshape(3, 10) + 1000
        self.simple_hilicp_data = pd.DataFrame(hilicp_data,
                                               index=hilicp_snames,
                                               columns=hilicp_fnames)

        sample_link = np.array(c18p_snames + c18n_snames + hilicp_snames)
        columns = ('c18positive', 'c18negative', 'hilicpositive')
        self.simple_sample_link = pd.DataFrame(sample_link.reshape(3,3).T,
                                               columns=columns)


    def test_simple(self):
        # test simple data
        exp_data = np.concatenate((np.arange(30).reshape(3, 10),
                                  (np.arange(30) + 100).reshape(3, 10),
                                  (np.arange(30) + 1000).reshape(3, 10)),
                                  axis=1).astype(float)
        fnames = np.concatenate((self.simple_c18p_data.columns,
                                 self.simple_c18n_data.columns,
                                 self.simple_hilicp_data.columns))
        simple_exp_data = pd.DataFrame(exp_data, columns=fnames)
        simple_exp_sample_link = self.simple_sample_link

        obs_data, obs_sample_link = join_across_modes(self.simple_sample_link,
                                                      self.simple_c18p_data,
                                                      self.simple_c18n_data,
                                                      self.simple_hilicp_data)

        pd.testing.assert_frame_equal(obs_data, simple_exp_data)
        pd.testing.assert_frame_equal(obs_sample_link, simple_exp_sample_link)


    def test_with_sample_link_nans(self):
        # This sample link matrix should remove a row
        sample_link = self.simple_sample_link.copy()
        sample_link['c18positive'][0] = np.nan
        sample_link.iloc[2] = np.nan

        exp_data = np.concatenate((np.arange(30).reshape(3, 10),
                                  (np.arange(30) + 100).reshape(3, 10),
                                  (np.arange(30) + 1000).reshape(3, 10)),
                                  axis=1).astype(float)
        exp_data = exp_data[:2, :]
        exp_data[0, 0:10] = np.nan
        fnames = np.concatenate((self.simple_c18p_data.columns,
                                 self.simple_c18n_data.columns,
                                 self.simple_hilicp_data.columns))
        exp_data = pd.DataFrame(exp_data, columns=fnames)
        exp_sample_link = sample_link.iloc[:2, :]

        obs_data, obs_sample_link = join_across_modes(sample_link,
                                                      self.simple_c18p_data,
                                                      self.simple_c18n_data,
                                                      self.simple_hilicp_data)

        pd.testing.assert_frame_equal(obs_data, exp_data)
        pd.testing.assert_frame_equal(obs_sample_link, exp_sample_link)

    def test_with_nans_in_both(self):
        # Test when sample_link and data have nans.
        # This sample link matrix should remove a row
        sample_link = self.simple_sample_link.copy()
        sample_link['c18negative'][1] = np.nan

        c18p_data = self.simple_c18p_data.copy()
        c18n_data = self.simple_c18n_data.copy()
        hilicp_data = self.simple_hilicp_data.copy()

        # nans for the missing sample, not strictly necessary since the fact
        # that it is missing in sample link will cast these entries to nans
        c18n_data.iloc[1] = np.nan
        # nans in the data from non-detects
        c18p_data.iloc[0, [4,6,7]] = np.nan
        c18p_data.iloc[2, [4,6]] = np.nan
        c18n_data.iloc[0, 1] = np.nan
        c18n_data.iloc[2, [1, 2]] = np.nan
        hilicp_data.iloc[[0, 2], 8] = np.nan

        exp_data = np.concatenate((np.arange(30).reshape(3, 10),
                                  (np.arange(30) + 100).reshape(3, 10),
                                  (np.arange(30) + 1000).reshape(3, 10)),
                                  axis=1).astype(float)
        # nans for missing sample
        exp_data[1, 10:20] = np.nan
        # nans from non-detects
        exp_data[0, [4,6,7,11,28]] = np.nan
        exp_data[2, [12, 11, 28, 4, 6]] = np.nan

        fnames = np.concatenate((self.simple_c18p_data.columns,
                                 self.simple_c18n_data.columns,
                                 self.simple_hilicp_data.columns))
        exp_data = pd.DataFrame(exp_data, columns=fnames)
        exp_sample_link = sample_link

        obs_data, obs_sample_link = join_across_modes(sample_link,
                                                      c18p_data,
                                                      c18n_data,
                                                      hilicp_data)

        pd.testing.assert_frame_equal(obs_data, exp_data)
        pd.testing.assert_frame_equal(obs_sample_link, exp_sample_link)


    def test_with_extraneous_sample_link_columns(self):
        sample_link = self.simple_sample_link.copy()
        sample_link['a'] = '235252'
        sample_link['b'] = 5
        sample_link[3] = 'sdgsdfg'
        sample_link['x'] = np.nan

        exp_data = np.concatenate((np.arange(30).reshape(3, 10),
                                  (np.arange(30) + 100).reshape(3, 10),
                                  (np.arange(30) + 1000).reshape(3, 10)),
                                  axis=1).astype(float)
        fnames = np.concatenate((self.simple_c18p_data.columns,
                                 self.simple_c18n_data.columns,
                                 self.simple_hilicp_data.columns))
        simple_exp_data = pd.DataFrame(exp_data, columns=fnames)
        simple_exp_sample_link = sample_link

        obs_data, obs_sample_link = join_across_modes(sample_link,
                                                      self.simple_c18p_data,
                                                      self.simple_c18n_data,
                                                      self.simple_hilicp_data)

        pd.testing.assert_frame_equal(obs_data, simple_exp_data)
        pd.testing.assert_frame_equal(obs_sample_link, simple_exp_sample_link)


class TestUtil__combine_features_in_fc_data(unittest.TestCase):
    def setUp(self):
        tmp = np.arange(40, dtype=float).reshape(5, 8)
        index = ['s%s' % i for i in range(5)]
        columns = ['f%s' % i for i in range(8)]
        self.data = pd.DataFrame(tmp, index=index, columns=columns)


    def test_errors(self):
        features_to_group = [(['f0', 'f1'], ['f1'], 3),
                             (['f2', 'f0'], ['f2'], 2)]
        new_names = ['f1_1']
        self.assertRaises(ValueError, combine_features_in_fc_data, self.data,
                          features_to_group, new_names)

        features_to_group = [(['f0', 'f1'], ['f1'], 3),
                             (['f2', 'f3'], ['f1'], 2)]
        new_names = ['f1_1', 'f2_1']
        self.assertRaises(ValueError, combine_features_in_fc_data, self.data,
                          features_to_group, new_names)

        features_to_group = [(['f0', 'f1'], ['f1'], 3),
                             (['f2', 'f3'], ['f2'], 2)]
        new_names = ['f1_1', 'f1_1']
        self.assertRaises(ValueError, combine_features_in_fc_data, self.data,
                          features_to_group, new_names)

        features_to_group = [(['f0', 'f1'], ['f1'], 3),
                             (['f2', 'f3'], ['fxyz'], 2)]
        new_names = ['f1_1', 'f2_1']
        self.assertRaises(ValueError, combine_features_in_fc_data, self.data,
                          features_to_group, new_names)

        features_to_group = [(['f0', 'f1'], ['f1'], 3),
                             (['f2', 'f3'], ['f2'], 2)]
        new_names = ['f1_1', 'f6']
        self.assertRaises(ValueError, combine_features_in_fc_data, self.data,
                          features_to_group, new_names)


    def test_simple(self):
        features_to_group = [(['f0','f1'], ['f1'], 4)]
        new_names = ['f1_1']
        pref_strength_threshhold = 3

        exp = self.data.drop(['f0', 'f1'], axis='columns')
        exp['f1_1'] = np.array([1, 9, 17, 25, 33], dtype=float)

        obs = combine_features_in_fc_data(self.data, features_to_group,
                                          new_names, pref_strength_threshhold)

        pd.testing.assert_frame_equal(obs, exp)


        features_to_group = [(['f0','f1'], ['f1'], 3)]
        new_names = ['f1_1']
        pref_strength_threshhold = 4

        exp = self.data.drop(['f0', 'f1'], axis='columns')
        exp['f1_1'] = np.array([0.5, 8.5, 16.5, 24.5, 32.5], dtype=float)

        obs = combine_features_in_fc_data(self.data, features_to_group,
                                          new_names, pref_strength_threshhold)
        
        pd.testing.assert_frame_equal(obs, exp)


        features_to_group = [(['f0', 'f1'], ['f1'], 3),
                             (['f3', 'f2'], ['f2'], 3),
                             (['f4', 'f5', 'f7'], ['f4'], 2)]
        new_names = ['f1_1', 'f2_1', 'f3_1']
        pref_strength_threshhold = 3
        obs = combine_features_in_fc_data(self.data, features_to_group,
                                          new_names, pref_strength_threshhold)

        exp = self.data.drop(['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f7'],
                             axis='columns')
        exp['f1_1'] = np.array([1, 9, 17, 25, 33], dtype=float)
        exp['f2_1'] = np.array([2, 10, 18, 26, 34], dtype=float)
        exp['f3_1'] = np.array([16/3., 40/3., 64/3., 88/3., 112/3.],
                               dtype=float)

        pd.testing.assert_frame_equal(obs, exp)


    def test_feature_with_nans(self):
        data = self.data.copy()
        data.iloc[[0, 1], [0, 0]] = np.nan
        data.iloc[[1, 2], [1, 1]] = np.nan

        features_to_group = [(['f0', 'f3', 'f1'], ['f1'], 4)]
        new_names = ['f1_1']
        pref_strength_threshhold = 4

        exp = self.data.drop(['f0', 'f3', 'f1'], axis='columns')
        exp['f1_1'] = np.array([1, 11, 17.5, 25, 33], dtype=float)

        obs = combine_features_in_fc_data(data, features_to_group,
                                          new_names, pref_strength_threshhold)

        pd.testing.assert_frame_equal(obs, exp)


    def test_multiple_pref_features(self):
        features_to_group = [(['f4', 'f5', 'f7'], ['f4', 'f5'], 3)]
        new_names = ['fx']
        pref_strength_threshhold = 3

        obs = combine_features_in_fc_data(self.data, features_to_group,
                                          new_names, pref_strength_threshhold)

        exp = self.data.drop(['f4', 'f5', 'f7'], axis='columns')
        exp['fx'] = np.array([4.5, 12.5, 20.5, 28.5, 36.5], dtype=float)
        
        pd.testing.assert_frame_equal(obs, exp)


        data = self.data.copy()
        data.iloc[[0, 1], [0, 0]] = np.nan
        data.iloc[[1, 2], [1, 1]] = np.nan

        features_to_group = [(['f1', 'f2', 'f0'], ['f0', 'f2'], 4)]
        new_names = ['fx']
        pref_strength_threshhold = 4

        exp = self.data.drop(['f0', 'f1', 'f2'], axis='columns')
        exp['fx'] = np.array([2, 10, 17, 25, 33], dtype=float)

        obs = combine_features_in_fc_data(data, features_to_group,
                                          new_names, pref_strength_threshhold)

        pd.testing.assert_frame_equal(obs, exp)


class TestCase__add_multipeaks(unittest.TestCase):
    def setUp(self):
        tmp = np.arange(40, dtype=float).reshape(5, 8)
        index = ['s%s' % i for i in range(5)]
        columns = ['f%s' % i for i in range(8)]
        self.data = pd.DataFrame(tmp, index=index, columns=columns)

    def test_simple(self):
        to_combine = [['f0', 'f1']]
        new_names = ['f_1+2']
        obs = add_multipeaks(self.data, to_combine, new_names)
        exp = self.data.drop(to_combine[0], axis='columns')
        exp[new_names[0]] = self.data[to_combine[0]].sum(1)
        pd.testing.assert_frame_equal(obs, exp)

        to_combine = [['f0', 'f1'], ['f4', 'f2']]
        new_names = ['f_0+1', 'f_2+4']
        obs = add_multipeaks(self.data, to_combine, new_names)
        exp = self.data.drop(superset(to_combine), axis='columns')
        exp[new_names[0]] = self.data[to_combine[0]].sum(1)
        exp[new_names[1]] = self.data[to_combine[1]].sum(1)
        pd.testing.assert_frame_equal(obs, exp)


    def test_errors(self):
        to_combine = [['f0', 'f1232']]
        new_names = ['fxxx']
        self.assertRaises(ValueError, add_multipeaks, self.data,
                          to_combine, new_names)

        to_combine = [['f0', 'f1']]
        new_names = ['f3']
        self.assertRaises(ValueError, add_multipeaks, self.data,
                          to_combine, new_names)


if __name__ == '__main__':
    unittest.main()