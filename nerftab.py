#%%
# Generating the allinone table
# Take the output of flatforest() to generate the network ready table for the NERF progress
# Author: Yue Zhang
# Contact: Yue.zhang@lih.lu
# Oct 2018

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time


def timing(func):
    def wrap(*args, **kw):
        print('<function name: {0}>'.format(func.func_name))
        time1 = time.time()
        ret = func(*args, **kw)
        time2 = time.time()
        print('[timecosts: {0} s]'.format(time2-time1))
        return ret
    return wrap


@timing
# nerftab function for generating pairs of the 'correct' decision features
def nerftab(fflist):
    # All possible pairs generator
    # TODO give the tree a index?
    # TODO Same to previous one, give sample a index to loop with
    list_of_single_decisions = pd.DataFrame()
    list_of_all_decision_pairs = pd.DataFrame()

    for psample in range(max(TIE[1].loc[:, 'sample_index'])):  # Loop on predict samples
        t_in_psample = TIE[2].loc[(TIE[2]['matching'] == 'match') & (TIE[2]['sample'] == psample),'tree index']
        # Trees with the 'correct' prediction during the prediction process of the sample psample

        t_p_psample = TIE[1].loc[(TIE[1]['tree_index'].isin(t_in_psample)) &
                                 (TIE[1]['sample_index'] == psample) & (t_p_psample['feature_index'] != -2),]
        # For each sample, get the 'correct' trees with decision paths, remove the leaf nodes

        single_decisions_sample = pd.DataFrame()
        all_decision_pairs_sample = pd.DataFrame()
        for itree in t_in_psample:
                nodes_cand = t_p_psample.loc[t_p_psample['tree_index'] == itree,]
                if nodes_cand.shape[0] == 1:
                #  If-statement for the situation where only one decision node presents
                    single_decisions_sample = pd.concat(single_decisions_sample, nodes_cand)
                    continue
                else:
                    pairsofonetree = pd.DataFrame()
                    for i in range(nodes_cand.shape[0]):
                        pairs_inside = pd.DataFrame()
                        for j in range(i + 1, nodes_cand.shape[0]):
                            f_i = nodes_cand.iloc[i, 0]  # feature index of feature i
                            f_j = nodes_cand.iloc[j, 0]
                            gs_i = nodes_cand.iloc[i, 1]  # gs of feature i
                            gs_j = nodes_cand.iloc[j, 1]
                            ft_i = nodes_cand.iloc[i, 3]  # feature threshold of i
                            ft_j = nodes_cand.iloc[j, 3]
                            tr_index = nodes_cand.iloc[i, 2]
                            sp_index = nodes_cand.iloc[i, 4]
                            listofunit = [[f_i, f_j, gs_i, gs_j, ft_i, ft_j, tr_index, sp_index]]
                            dfunit = pd.DataFrame(listofunit,
                                                  columns=['feature_i', 'feature_j', 'GS_i', 'GS_j', 'threshold_i',
                                                           'threshold_j',
                                                           'tree_index', 'sample_index'])
                            pairs_inside = pd.concat([pairs_inside, dfunit])
                        pairsofonetree = pd.concat([pairsofonetree, pairs_inside])


#     #%%
#     max(raw_hits.loc[:, 'sample_index'])  # Number of samples
#     for s_in in range(max(raw_hits.loc[:, 'sample_index'])):
#         single_tree = df[1].loc[df[1]['sample_index' == s_in],]  # TODO columns add or not?
#         for t_in in range(max(single_tree))
#
#     max(raw_hits.loc[:, 'tree_index'])  # 200 trees in the test case
#     for n in range(max(raw_hits.loc[:, 'tree_index'])):
#
#     max(raw_hits.iloc)
#     for mm in range(max(raw_hits.loc)):
# #%%
# for i in TIE[2].loc[(TIE[2]['matching'] == 'match') & (TIE[2]['sample'] == 1),'tree index']:
#     print(i)
# #%%
# t_in_psample = TIE[2].loc[(TIE[2]['matching'] == 'match') & (TIE[2]['sample'] == 1),'tree index']
# # Trees with the 'correct' prediction during the prediction process of the sample psample
#
# t_p_psample = TIE[1].loc[(TIE[1]['tree_index'].isin(t_in_psample)) & (TIE[1]['sample_index'] == 1),]
# nodes_cand = t_p_psample.loc[(t_p_psample['tree_index'] == 3) & (t_p_psample['feature_index'] != -2),]
# nodes_cand.loc[:, 'feature_index']
# hit['sample_index'] = pd.Series(s_index).values
# ft_i = nodes_cand.iloc[0, 3]
# nodes_cand.iloc[0, 0]
#
# #
# pairsofonetree = pd.DataFrame()
# for i in range(nodes_cand.shape[0]):
#     pairs_inside = pd.DataFrame()
#     for j in range(i+1, nodes_cand.shape[0]):
#         f_i = nodes_cand.iloc[i, 0]  # feature index of feature i
#         f_j = nodes_cand.iloc[j, 0]
#         gs_i = nodes_cand.iloc[i, 1]  # gs of feature i
#         gs_j = nodes_cand.iloc[j, 1]
#         ft_i = nodes_cand.iloc[i, 3]  # feature threshold of i
#         ft_j = nodes_cand.iloc[j, 3]
#         tr_index = nodes_cand.iloc[i, 2]
#         sp_index = nodes_cand.iloc[i, 4]
#         listofunit = [[f_i, f_j, gs_i, gs_j, ft_i, ft_j, tr_index, sp_index]]
#         dfunit = pd.DataFrame(listofunit, columns = ['feature_i', 'feature_j', 'GS_i', 'GS_j', 'threshold_i', 'threshold_j',
#                                                      'tree_index', 'sample_index'])
#         pairs_inside = pd.concat([pairs_inside, dfunit])
#     pairsofonetree = pd.concat([pairsofonetree, pairs_inside])
