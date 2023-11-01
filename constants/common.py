import datetime

import pandas as pd


START_HISTORICAL_DATE = datetime.datetime(2006, 1, 1)


def calc_drawdown(ret: pd.Series):
    term = pd.DataFrame({
        "ret": ret.dropna().cumsum(),
        'lagged_ret': ret.dropna().cumsum().shift(1),
        'high_watermark_index': ret.dropna().cumsum().index
    })
    term['lagged_ret'] = term['lagged_ret'].fillna(0)

    highest_watermark_index = term.iloc[0]['high_watermark_index']
    for d in term.index:
        row = term.loc[d]
        if row['ret'] >= term.loc[highest_watermark_index]['ret']:
            highest_watermark_index = d
        term.loc[d, 'high_watermark_index'] = highest_watermark_index

    term['high_watermark_ret'] = term.apply(
        lambda x: term.loc[x['high_watermark_index']]['ret'], axis=1)
    term['drawdown_deep'] = term['high_watermark_ret'] - term['ret']
    term['drawdown_duration'] = term.index - term['high_watermark_index']

    max_drawdown_deep = term.loc[term['drawdown_deep'].idxmax()]
    max_drawdown_duration = term.loc[term['drawdown_duration'].idxmax()]
    return max_drawdown_deep, max_drawdown_duration
