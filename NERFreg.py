#%%
# calculating the I(Fi, Fj) while generating the edge intensity
# Take the output of nerftab() to generate the network ready table for the NERF progress
# Author: Yue Zhang
# Contact: Yue.zhang@lih.lu
# Oct 2018

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time


def timing(func):
    def wrap(*args, **kw):
        print('<function name: {0}>'.format(func.__name__))
        time1 = time.time()
        ret = func(*args, **kw)
        time2 = time.time()
        print('[timecosts: {0} s]'.format(time2-time1))
        return ret
    return wrap


@timing
def localnerf(nf_ff, local_index):
    # TODO Try get this thing out before 3:00 pm, if so, make an appointment with FA on Wednesday,
    # TODO and make the first network of it in the afternoon
#%%
local_index = 0

allpairs_local = YOLO[1].loc[(YOLO[1]['sample_index'] == local_index, )]
allpairs_local['GSP'] = (allpairs_local.loc[:, 'GS_i'] + allpairs_local.loc[:, 'GS_j'])
test1 = pd.Series(map(frozenset, zip(allpairs_local['feature_i'],allpairs_local['feature_j']))).value_counts().reset_index()
test1.rename(columns={'index':'places',0:'count'}, inplace=True)

# TODO replace frozenset as soon as possible since it need a O(N2) loop later..
# Create a frozenset column, with the two features
allpairs_local['pairs'] = list(map(frozenset, zip(allpairs_local['feature_i'], allpairs_local['feature_j'] )))

ei = list()
# Loop on unique pairs
for pp in range(test1.shape[0]):
    uniquepair = test1.iloc[pp, 0]
    upcounts = test1.iloc[pp, 1]
    gspsum = 0
    counterofhits = 0
    #  Loop on the ref list
    # TODO change frozenset to groupby later, this O(N*N) is slow like crazy
    for ap in range(allpairs_local.shape[0]):
        if allpairs_local.iloc[ap, 9] == uniquepair:
            gsp = allpairs_local.iloc[ap, 8]
            gspsum = gspsum + gsp
            counterofhits += 1
    gspba = gspsum/counterofhits
    eisingle = gspba * upcounts
    ei.append(eisingle)

test1['EI'] = pd.Series(ei).values

    #%%
    uniquepair = test1.iloc[1, 0]
