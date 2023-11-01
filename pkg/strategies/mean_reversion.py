from typing import List
import numpy as np
import streamlit as st
import math
import plotly.figure_factory as ff
from scipy import linalg
from scipy import stats
from pykalman import KalmanFilter

import pandas as pd
from constants.common import START_HISTORICAL_DATE

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from pkg.data_provider.wrapper import DataProviderWrapper
from pkg.instrument.model import Instrument

from constants.common import calc_drawdown


class MeanReversionStrategy:
    __data_provider: DataProviderWrapper
    __LOOK_BACK: int = 46  # TODO: calc this value on training data
    __ENTRY_ZSCORE: float = 2.0
    __EXIT_ZSCORE: float = 0.5

    def __init__(self, data_provider: DataProviderWrapper) -> None:
        self.__data_provider = data_provider

    def __get_hedge_ratio(self, price: pd.DataFrame) -> pd.DataFrame:
        """
            Calculate hedge ratio
        """
        hedge_ratio = pd.DataFrame(
            np.NaN, columns=price.columns, index=price.index)
        for i in range(self.__LOOK_BACK-1, len(price)):
            price_period = price[i-self.__LOOK_BACK+1:i+1]
            jres = coint_johansen(price_period, det_order=0, k_ar_diff=1)
            hedge_ratio.iloc[i] = jres.evec.T[0]
        return hedge_ratio

    def calc_trading_inputs(self, price: pd.DataFrame):
        observed_column, remaining_columns = price.columns[0], price.columns[1:]
        num_data_points, num_states = len(price), len(remaining_columns) + 1
        measurement_vectors = price[observed_column].values
        observation_matrices = np.concatenate(
            [price[remaining_columns].values, np.ones((len(price), 1))], axis=1).reshape((num_data_points, 1, num_states))
        # State transition matrix
        F = np.eye(num_states)
        # Initialize estimated state, estimated state variance
        est_states = np.empty((num_data_points, num_states))
        est_state_covariances = np.empty(
            (num_data_points, num_states, num_states))
        # There is a method to estimate these variances from data
        # For keep simplicity, pick theses as author mentioned
        delta = 0.0001
        process_noise_err = delta / (1 - delta) * np.eye(num_states)
        measurement_err = 0.001

        pred_measurement_covariances = np.empty(
            (num_data_points, 1, 1))
        pred_measurement_errors = np.empty((num_data_points, 1))
        for i in range(num_data_points):
            # State extrapolation from t-1
            pred_state = F.dot(
                est_states[i-1]) if i > 0 else np.zeros(num_states)
            # Covariance extrapolation from t-1
            pred_state_cov = F.dot(est_state_covariances[i-1]).dot(
                F.T) + process_noise_err if i > 0 else np.zeros((num_states, num_states))

            # Measurement prediction from t - 1
            pred_measurement = np.matmul(
                observation_matrices[i], pred_state)
            # Measurement variance prediction from t - 1
            pred_measurement_covariances[i] = observation_matrices[i].dot(
                pred_state_cov).dot(observation_matrices[i].T) + measurement_err
            # Measurement at t
            pred_measurement_errors[i] = measurement_vectors[i] - \
                pred_measurement
            # Kalman Gain
            K = pred_state_cov.dot(observation_matrices[i].T.dot(
                linalg.pinv(pred_measurement_covariances[i])))
            # State update
            est_states[i] = pred_state + K.dot(pred_measurement_errors[i])
            # State variance update
            est_state_covariances[i] = pred_state_cov - \
                K.dot(observation_matrices[i].dot(pred_state_cov))
        hedge_ratio = np.concatenate([np.ones((num_data_points, 1)),  est_states * -1],
                                     axis=1)[:, :len(price.columns)]
        hedge_ratio[:2].fill(np.NaN)
        pred_measurement_errors[:2].fill(np.NaN)
        pred_measurement_covariances[:2].fill(np.NaN)
        hedge_ratio = pd.DataFrame(
            hedge_ratio, columns=price.columns, index=price.index)

        spread = pd.Series(pred_measurement_errors[:, 0], index=price.index)
        spread_std = pd.Series(
            np.sqrt(pred_measurement_covariances[:, 0, 0]), index=price.index)
        return hedge_ratio, spread, spread_std

    def trade(self, price: pd.DataFrame, hedge_ratio: pd.DataFrame, spread: pd.Series, spread_std: pd.Series):
        long_entry_signal = spread < -spread_std
        long_exit_signal = spread >= -spread_std
        long_num_units = pd.Series(np.NaN, index=price.index)
        long_num_units.iloc[0] = 0
        long_num_units[long_entry_signal] = 1
        long_num_units[long_exit_signal] = 0
        long_num_units = long_num_units.fillna(method='ffill')

        short_entry_signal = spread > spread_std
        short_exit_signal = spread <= spread_std
        short_num_units = pd.Series(np.NaN, index=price.index)
        short_num_units.iloc[0] = 0
        short_num_units[short_entry_signal] = -1
        short_num_units[short_exit_signal] = 0
        short_num_units = short_num_units.fillna(method='ffill')

        num_units = long_num_units + short_num_units
        if (num_units == 0).all():
            raise ValueError("Not have any entry signal")
        position = price * hedge_ratio.mul(num_units, axis=0)
        pnl = (price - price.shift(1))/price.shift(1) * position.shift(1)
        pnl = pnl.sum(axis=1)
        ret = pnl / position.shift(1).sum(axis=1).abs()
        sharp_ratio = math.sqrt(252)*ret.mean()/ret.std()
        max_drawdown_deep, max_drawdown_duration = calc_drawdown(ret)
        # Halflife
        # Annual percentage rate
        return ret, sharp_ratio, max_drawdown_deep, max_drawdown_duration

    def generate_trading_signal(self, instruments: List[Instrument]):
        price_map = {}
        for instrument in instruments:
            historical_data = self.__data_provider.load_historical_data(
                instrument=instrument, start_date=START_HISTORICAL_DATE
            )
            price_map[str(instrument)] = historical_data['close']
        price = pd.DataFrame(price_map).sort_index()
        hedge_ratio = self.__get_hedge_ratio(price=price)
        mkt_value = price * hedge_ratio
        spread = mkt_value.sum(axis=1)
        zscore_spread = (spread - spread.rolling(self.__LOOK_BACK).mean()
                         ) / spread.rolling(self.__LOOK_BACK).std()

        long_entry_signal = zscore_spread < -self.__ENTRY_ZSCORE
        long_exit_signal = zscore_spread >= -self.__EXIT_ZSCORE
        long_num_units = pd.Series(np.NaN, index=zscore_spread.index)
        long_num_units.iloc[0] = 0
        long_num_units[long_entry_signal] = 1
        long_num_units[long_exit_signal] = 0
        long_num_units = long_num_units.fillna(method='ffill')

        short_entry_signal = zscore_spread > self.__ENTRY_ZSCORE
        short_exit_signal = zscore_spread <= self.__EXIT_ZSCORE
        short_num_units = pd.Series(np.NaN, index=zscore_spread.index)
        short_num_units.iloc[0] = 0
        short_num_units[short_entry_signal] = -1
        short_num_units[short_exit_signal] = 0
        short_num_units = short_num_units.fillna(method='ffill')

        num_units = long_num_units + short_num_units
        return zscore_spread, hedge_ratio.mul(num_units, axis=0)

    def backtest1(self, instruments: List[Instrument]):
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import TimeSeriesSplit
        # get historical data
        # split in-sample/out-sample dataset
        # split in-sample dataset into folds
        # each fold
        #   split train, validation dataset
        #   get trading params from train dataset
        #   run backtest on validation by using trading params to get unlevered returns
        #       when having enough unlevered returns, calculate levered returns

        price_map = {}
        for instrument in instruments:
            historical_data = self.__data_provider.load_historical_data(
                instrument=instrument, start_date=START_HISTORICAL_DATE
            )
            price_map[str(instrument)] = historical_data['close']
        all_price = pd.DataFrame(price_map).sort_index()

        in_sample_dataset, out_sample_dataset = train_test_split(
            all_price, test_size=0.3, shuffle=False)

        tscv = TimeSeriesSplit(n_splits=4)
        # Each fold
        st.header("In-sample dataset")
        arr_halflife = []
        arr_ret = []
        arr_jres = []
        arr_zscore_spread = []
        for i, fold in enumerate(tscv.split(in_sample_dataset)):
            # Training step
            train_indices, valid_indices = fold
            train_data, valid_data = in_sample_dataset.iloc[
                train_indices], in_sample_dataset.iloc[valid_indices]
            jres, zscore_spread, halflife = get_trading_params(train_data)
            arr_halflife.append(halflife)
            arr_jres.append(jres)
            arr_zscore_spread.append(zscore_spread)
            # Validation step
            lookback = round(halflife)
            # I think I need to handle the period that it doesn't too short
            # I will do that later
            ret = get_trading_result(lookback, valid_data)
            arr_ret.append(ret)

        weights = []
        for i, tab in enumerate(st.tabs([f'Fold {i+1}' for i in range(len(arr_ret))])):
            tab.subheader("Training sample")
            zscore_spread = arr_zscore_spread[i]
            tab.text(
                f"Time range: {zscore_spread.index.min()} - {zscore_spread.index.max()}")
            tab.text(f"{zscore_spread.index.max() - zscore_spread.index.min()}")
            tab.text(f"Half life: {arr_halflife[i]}")
            jres = arr_jres[i]
            tab.text(f"Eigen statistic: \n\t{jres.max_eig_stat}")
            tab.text(
                f'Eigen statistic critical: \n\t{jres.max_eig_stat_crit_vals}')
            tab.text("Zscore Spread")
            tab.line_chart(zscore_spread)

            tab.subheader("Validation sample")
            ret = arr_ret[i]
            tab.text(f"Time range: {ret.index.min()} - {ret.index.max()}")
            tab.text(f"{ret.index.max() - ret.index.min()}")

            tab.text(f"Return \n\t{ret.describe()}")
            sharp_ratio = math.sqrt(252)*ret.mean()/ret.std()
            weights.append(max(sharp_ratio, 0))
            tab.text(f"Sharp ratio: {sharp_ratio}")
            tab.line_chart(data=ret.cumsum().dropna())
        # Out-sample dataset
        st.header("Out sample")
        weighted_avg_halflife = (
            pd.Series(weights) * pd.Series(arr_halflife)).sum() / pd.Series(weights).sum()
        st.text(f"Weighted average halflife: {weighted_avg_halflife}")
        ret = get_trading_result(
            round(weighted_avg_halflife), out_sample_dataset)
        h = get_hedge_ratio(out_sample_dataset)
        st.write(h[2:].describe())
        st.line_chart(h[2:])
        get_hedge_ratio_v2(out_sample_dataset)
        fig = ff.create_distplot(
            [h[1:]['spread'], h[1:]['std']], ['spread', 'std'])
        st.plotly_chart(fig, use_container_width=True)
        st.text(f"Time range: {ret.index.min()} - {ret.index.max()}")
        st.text(f"Weighted average halflife: {weighted_avg_halflife}")
        st.text(f"Return \n\t{ret.describe()}")
        st.line_chart(ret.cumsum().dropna())
        st.text(f"Sharp ratio: {math.sqrt(252)*ret.mean()/ret.std()}")
        a = ret.dropna() + 1
        st.text(
            f"Geometric mean: {np.exp(np.log(a.prod())/a.notna().sum())}")
        leverage_ratio = get_leverage_ratio(ret.dropna())

        st.write(leverage_ratio)


def get_leverage_ratio(ret: pd.Series):
    leverage_lookback = 20 * 6
    base_equity = 1000
    result_df = pd.DataFrame(np.NaN, index=ret.index, columns=[
                             'ratio', 'equity', 'capital', 'ret'])
    for i in range(leverage_lookback, len(ret)):
        r = ret[i-leverage_lookback:i]
        ratio = r.mean() / (r.std())**2
        lagged_row = result_df.iloc[i-1]
        equity = (lagged_row['equity'] + lagged_row['ret'] * lagged_row['capital']) if not math.isnan(
            lagged_row['equity']) else base_equity
        capital = equity * ratio
        result_df.iloc[i]['ratio'] = ratio
        result_df.iloc[i]['equity'] = equity
        result_df.iloc[i]['capital'] = capital
        result_df.iloc[i]['ret'] = ret[i]

    return result_df


def get_hedge_ratio_v2(price: pd.DataFrame):
    observed_column, remaining_columns = price.columns[0], price.columns[1:]
    num_data_points, num_states = len(price), len(remaining_columns) + 1
    obs_matrices = np.concatenate(
        [price[remaining_columns].values, np.ones(price[remaining_columns].shape)], axis=1).reshape((num_data_points, 1, num_states))
    delta = 0.0001
    trans_cov = delta / (1 - delta) * np.eye(num_states)
    kf = KalmanFilter(
        # n_dim_obs=(1, 1),
        # n_dim_state=(num_states, 1),
        initial_state_mean=np.zeros(num_states),
        initial_state_covariance=np.zeros((num_states, num_states)),
        transition_matrices=np.eye(num_states),
        observation_matrices=obs_matrices,
        observation_covariance=0.001,
        observation_offsets=0,
        transition_offsets=None,
        transition_covariance=trans_cov
    )
    measurement_vectors = price[observed_column].values
    state_means, state_covs = kf.filter(measurement_vectors)
    print(state_means)


def get_hedge_ratio(price: pd.DataFrame):
    observed_column, remaining_columns = price.columns[0], price.columns[1:]
    num_data_points, num_states = len(price), len(remaining_columns) + 1
    measurement_vectors = price[observed_column].values
    obs_matrices = np.concatenate(
        [price[remaining_columns].values, np.ones(price[remaining_columns].shape)], axis=1).reshape((num_data_points, 1, num_states))
    # State transition matrix
    F = np.eye(num_states)
    # Initialize estimated state, estimated state variance
    est_state = np.zeros(num_states)
    est_state_cov = np.zeros((num_states, num_states))
    # Error
    # There is a method to estimate these variances from data
    # For keep simplicity, pick theses as author mentioned
    delta = 0.0001
    process_noise_err = delta / (1 - delta) * np.eye(num_states)
    measurement_err = 0.001

    est_states = []
    pred_measurement_covs = []
    errors = []

    for i in range(num_data_points):
        if i == 0:
            pred_state = est_state
            pred_state_cov = est_state_cov
        else:
            # State extrapolation from t-1
            pred_state = F.dot(est_state)
            # Covariance extrapolation from t-1
            pred_state_cov = F.dot(est_state_cov).dot(F.T) + process_noise_err

        # Measurement prediction from t - 1
        pred_measurement = np.matmul(obs_matrices[i], pred_state)
        # Measurement variance prediction from t - 1
        pred_measurement_cov = obs_matrices[i].dot(
            pred_state_cov).dot(obs_matrices[i].T) + measurement_err
        # Measurement at t
        pred_measurement_err = measurement_vectors[i] - pred_measurement
        # Kalman Gain
        K = pred_state_cov.dot(obs_matrices[i].T.dot(
            linalg.pinv(pred_measurement_cov)))
        # State update
        est_state = pred_state + K.dot(pred_measurement_err)
        # State variance update
        est_state_cov = pred_state_cov - \
            K.dot(obs_matrices[i].dot(pred_state_cov))
        est_states.append(est_state)
        pred_measurement_covs.append(pred_measurement_cov)
        errors.append(pred_measurement_err)
    est_states = np.array(est_states)
    errors = np.array(errors)
    print(est_states)

    pred_measurement_covs = np.array(pred_measurement_covs)
    r = pd.DataFrame({
        'spread': errors[:, 0],
        'std': np.sqrt(pred_measurement_covs[:, 0, 0])
    })
    # r = pd.Series(errors[:, 0, 0], index=price.index)
    return r


def get_trading_result(lookback, price):
    entry_zscore = 1.5
    exit_zscore = 0.5
    hedge_ratio = pd.DataFrame(
        np.NaN, columns=price.columns, index=price.index)
    for i in range(lookback-1, len(price)):
        price_period = price[i-lookback+1:i+1]
        jres = coint_johansen(price_period, det_order=0, k_ar_diff=1)
        hedge_ratio.iloc[i] = jres.evec.T[0]
    market_val = hedge_ratio * price
    spread = market_val.sum(axis=1)
    zscore_spread = (spread - spread.rolling(lookback).mean()
                     ) / spread.rolling(lookback).std()

    long_entry_signal = zscore_spread < -entry_zscore
    long_exit_signal = zscore_spread >= -exit_zscore
    long_num_units = pd.Series(np.NaN, index=zscore_spread.index)
    long_num_units.iloc[0] = 0
    long_num_units[long_entry_signal] = 1
    long_num_units[long_exit_signal] = 0
    long_num_units = long_num_units.fillna(method='ffill')

    short_entry_signal = zscore_spread > entry_zscore
    short_exit_signal = zscore_spread <= exit_zscore
    short_num_units = pd.Series(np.NaN, index=zscore_spread.index)
    short_num_units.iloc[0] = 0
    short_num_units[short_entry_signal] = -1
    short_num_units[short_exit_signal] = 0
    short_num_units = short_num_units.fillna(method='ffill')

    num_units = long_num_units + short_num_units
    positions = hedge_ratio.mul(num_units, axis=0) * price
    pnl = positions.shift(1) * (price - price.shift(1))/price.shift(1)
    pnl = pnl.sum(axis=1)
    ret = pnl / positions.shift(1).sum(axis=1).abs()
    return ret


def get_trading_params(train_data: pd.DataFrame):
    import statsmodels.api as sm
    jres = coint_johansen(train_data, det_order=0, k_ar_diff=1)
    w1 = jres.evec.T[0]
    weights = pd.DataFrame([w1] * len(train_data),
                           columns=train_data.columns, index=train_data.index)
    spread = (weights * train_data).sum(axis=1)
    x = spread.shift().dropna()
    y = spread - spread.shift()
    y = y.dropna()
    x = sm.add_constant(x)
    results = sm.OLS(y, x).fit()
    halflife = -math.log(2) / results.params[0]
    zscore_spread = (spread - spread.mean()) / spread.std()
    return jres, zscore_spread, halflife
