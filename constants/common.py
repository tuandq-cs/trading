import datetime
import math
import numpy as np

import pandas as pd


START_HISTORICAL_DATE = datetime.datetime(2006, 1, 1)


def calc_sharp_ratio(ret: pd.Series):
    return math.sqrt(252)*ret.mean()/ret.std()


def calc_drawdown(ret: pd.Series):
    term = pd.DataFrame({
        "ret": ret.dropna().cumsum(),
        'high_watermark_index': ret.dropna().cumsum().index,
        'high_watermark_ret': np.NaN,
        'drawdown_deep': np.NaN,
        'drawdown_duration': np.NaN
    })
    if len(term) == 0:
        return term

    highest_watermark_index = None
    for d in term.index:
        row = term.loc[d]
        if not highest_watermark_index or row['ret'] >= term.loc[highest_watermark_index]['ret']:
            highest_watermark_index = d
        term.loc[d, 'high_watermark_index'] = highest_watermark_index
    term['high_watermark_ret'] = term['high_watermark_index'].transform(
        lambda x: term.loc[x]['ret'], axis=0)

    term['drawdown_deep'] = term['high_watermark_ret'] - term['ret']
    term['drawdown_duration'] = term.index - term['high_watermark_index']
    return term
