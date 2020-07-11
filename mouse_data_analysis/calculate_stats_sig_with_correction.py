import pandas as pd
from argparse import ArgumentParser
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

"""
Get p-values and corrected p-values for fold change values in a set of colonizations
relative to the germ-free values in the same sample type

python3 calculate_ttest_filtered_stats_sig_with_correction.py -f fold_change_matrix_mode_collapsed.txt --metadata metadata.txt --sample_type urine
"""

def calculate_pvalues(fc_matrix, colonizations, sample_type):
    #print(f"Sample type: {sample_type}\n")

    grouped = fc_matrix.groupby(['colonization'])

    # print(fc_matrix)

    # print(grouped.groups)

    control_group = fc_matrix[(fc_matrix['colonization'] == 'germ-free') & (fc_matrix['sample_type'] == sample_type) & (fc_matrix['experiment'] != 'conventional')]
    # print('\nControl group:\n')
    # print(control_group)

    metabolites = []
    df_colonizations = []
    fc_values = []
    pvalues = []

    for metabolite in fc_matrix.columns:
        if metabolite in ['experiment','colonization', 'sample_type']:
            continue

        for colonization in colonizations:
            exp_group = grouped.get_group(colonization)

            exp_group_values = exp_group[metabolite].values
            control_group_values = control_group[metabolite].values

            if np.isnan(exp_group_values).all() or np.isnan(control_group_values).all():
                #print("Values for metabolite {} are all nans's. Skipping...".format(metabolite))
                continue

            control_group_values = control_group_values[~np.isnan(control_group_values)]

            if np.all(exp_group_values == 0) and np.all(control_group_values == 0):
                #print("Values for metabolite {} are all 1's. Skipping...".format(metabolite))
                continue

            # print(f'metabolite: {metabolite}')
            # print('exp_group_values')
            # print(exp_group_values)
            # print('control_group_values')
            # print(control_group_values)

            # Perform Student's t-test
            statistic, pvalue = stats.ttest_ind(exp_group_values, control_group_values)

            metabolites.append(metabolite)
            df_colonizations.append(colonization)
            fc_values.append(np.nanmean(exp_group_values))
            pvalues.append(pvalue)

    # Perform Benjamini-Hochberg corrections for multiple comparisons
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(pvalues, method='fdr_bh')

    return pd.DataFrame({
        'colonization': df_colonizations,
        'fc_value_avg': fc_values,
        'pvalue': pvalues,
        'pvalue_corrected': pvals_corrected
    }, index=metabolites)


if __name__ == "__main__":
    parser = ArgumentParser(description='Generate p-values and corrected p-values')
    parser.add_argument('-f', '--fc', help="Path to mode-collapsed fold change matrix", required=True)
    parser.add_argument('--metadata', help="Path to metadata file", required=True)
    parser.add_argument('--sample_type', help="Sample type to look at", required=True)

    args = parser.parse_args()

    colonizations = ['Bt', 'Cs', 'Cs_Bt_Ca_Er_Pd_Et']
    sample_type = args.sample_type

    fc_matrix = pd.read_csv(args.fc, index_col=0, sep='\t')
    metadata = pd.read_csv(args.metadata, index_col=0, sep='\t')

    fc_matrix = fc_matrix \
        .join(metadata[['experiment','sample_type', 'colonization']])

    fc_matrix = fc_matrix[fc_matrix['colonization'].isin(colonizations + ['germ-free']) & (fc_matrix['sample_type'] == sample_type)]

    print(fc_matrix)

    df = calculate_pvalues(fc_matrix, colonizations, sample_type)

    print('p-value table:')
    print(df)

    df.to_excel(f"corrected_ttest_pvalues_{sample_type}.xlsx")
