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
def nerfreg():
    # TODO Try get this thing out before 3:00 pm, if so, make an appointment with FA on Wednesday,
    # TODO and make the first network of it in the afternoon
