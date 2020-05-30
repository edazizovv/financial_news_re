import pandas
import pandas_datareader
import datetime
import itertools
import numpy

# https://github.com/Fatal1ty/tinkoff-api


import asyncio
from tinkoff.investments import (
    TinkoffInvestmentsRESTClient, Environment, CandleResolution
)
from tinkoff.investments.client.exceptions import TinkoffInvestmentsError


def load_old(api_key, target_quotes, news_horizon, effect_horizon, max_quotes_lag, show_shapes=False, news_show=False):
    d = './data/data/rex.xlsx'
    data = pandas.read_excel(d)

    newstitle_frame = data[['id', 'time', 'title']]
    lag_markers = list(itertools.product(newstitle_frame['id'].values, numpy.array(numpy.arange(news_horizon - 1)) + 1))
    lag_markers = pandas.DataFrame(data=lag_markers, columns=['id', 'lag'])
    newstitle_frame = newstitle_frame.merge(right=lag_markers, left_on=['id'], right_on=['id'])

    newstitle_frame['target_date'] = newstitle_frame.apply(func=add_lag, axis=1)
    beginning_date, ending_date = newstitle_frame['target_date'].min() - datetime.timedelta(
        days=(max_quotes_lag + effect_horizon)), newstitle_frame['target_date'].max()

    the_batches = []
    for target_quote in target_quotes:
        f = pandas_datareader.av.time_series.AVTimeSeriesReader(symbols=target_quote, function='TIME_SERIES_DAILY',
                                                                start=beginning_date, end=ending_date,
                                                                # retry_count=3, pause=0.1, session=None, chunksize=25,
                                                                api_key=api_key)

        ff = f.read()
        nn = ff.shape[1]
        ff['ticker'] = target_quote
        the_batches.append(ff)
    quotes_data = pandas.concat(the_batches)
    quotes_data = quotes_data.sort_index(ascending=True)
    if show_shapes:
        print(quotes_data['open'].value_counts().shape)

    quotes_data_lagged_values, quotes_data_lagged_columns = lag(array=quotes_data.values,
                                                                names=quotes_data.columns.values,
                                                                exactly=effect_horizon, appx='hori', ex=['ticker'])
    quotes_data = pandas.DataFrame(data=quotes_data_lagged_values, index=quotes_data.index.values,
                                   columns=quotes_data_lagged_columns)
    if show_shapes:
        print(quotes_data['open_hori1'].value_counts().shape)

    for j in range(nn):
        quotes_data.iloc[:, 5 + j] = quotes_data.iloc[:, 5 + j] / quotes_data.iloc[:, j] - 1
    quotes_data = quotes_data.drop(columns=[quotes_data.columns.values[j] for j in range(nn)])

    quotes_data = quotes_data.dropna()
    if show_shapes:
        print(quotes_data['open_hori1'].value_counts().shape)

    quotes_data_lagged_values, quotes_data_lagged_columns = lag(array=quotes_data.values,
                                                                names=quotes_data.columns.values, upon=max_quotes_lag,
                                                                ex=['ticker'])
    quotes_data_lagged = pandas.DataFrame(data=quotes_data_lagged_values, index=quotes_data.index.values,
                                          columns=quotes_data_lagged_columns)

    quotes_data_lagged = quotes_data_lagged.dropna()
    if show_shapes:
        print(quotes_data_lagged['open_hori1_LAG0'].value_counts().shape)

    quotes_data_lagged = quotes_data_lagged.reset_index()
    quotes_data_lagged['index'] = quotes_data_lagged['index'].apply(func=to_date)
    the_data = quotes_data_lagged.merge(right=newstitle_frame, left_on='index', right_on='target_date')
    print(newstitle_frame['title'].value_counts().shape)
    if show_shapes:
        print(the_data['open_hori1_LAG0'].value_counts().shape)

    return the_data


async def show_my_time_candles(ticker, token, start_date, end_date, interval=CandleResolution.MIN_1):
    try:
        async with TinkoffInvestmentsRESTClient(
                # token='TOKEN',
                token=token,
                environment=Environment.SANDBOX) as client:

            instruments = await client.market.instruments.search(ticker)
            instrument = instruments[0]
            figi = instrument.figi

            candles = await client.market.candles.get(
                # figi='BBG000B9XRY4',
                # figi=ticker,
                figi=figi,
                # dt_from=datetime(2019, 1, 1),
                dt_from=start_date,
                # dt_to=datetime(2019, 12, 31),
                dt_to=end_date,
                interval=interval
            )
            data = []
            if len(candles) == 0:
                data = pandas.DataFrame(columns=['time', 'open', 'close', 'high', 'low', 'volume'])
            else:
                for candle in candles:
                    data.append(
                        pandas.DataFrame(data=[[candle.time, candle.o, candle.c, candle.h, candle.l, candle.v]]))
                    # print(f'{candle.time}: {candle.h}')
                data = pandas.concat(data, axis=0)
                data.columns = ['time', 'open', 'close', 'high', 'low', 'volume']
        return data
    except TinkoffInvestmentsError as e:
        print(e)


async def min_partitor(ticker, start_date, end_date, token):
    diff = end_date - start_date
    partitions_needed = diff > datetime.timedelta(days=1)
    if partitions_needed:
        result = []
        for dd in range(diff.days):
            # print(dd)
            start_part = start_date + datetime.timedelta(days=dd)
            end_part = start_date + datetime.timedelta(days=(dd + 1))
            # print(start_part)
            # print(end_part)
            resy = await show_my_time_candles(ticker=ticker, start_date=start_part, end_date=end_part, token=token)
            result.append(resy)
        start_part = start_date + datetime.timedelta(days=diff.days)
        end_part = end_date
        if end_part - start_part >= datetime.timedelta(minutes=1):
            resy = await show_my_time_candles(ticker=ticker, start_date=start_part, end_date=end_part, token=token)
            result.append(resy)
        result = pandas.concat(result, axis=0)
    else:
        result = await show_my_time_candles(ticker=ticker, start_date=start_date, end_date=end_date, token=token)
    return result


async def call_them_all(tickers, start_date, end_date, token):
    result = []
    for ticker in tickers:
        resy = await min_partitor(ticker=ticker, start_date=start_date, end_date=end_date, token=token)
        resy['ticker'] = ticker
        result.append(resy)
    result = pandas.concat(result, axis=0)

    return result


"""
Example of use

tickers = ['AAPL', 'MSFT', 'INTC']
start_date, end_date = datetime(2020, 1, 1), datetime(2020, 3, 1)

result = await call_them_all(tickers=tickers, start_date=start_date, end_date=end_date, token=token)
"""


async def load(api_key, target_quotes, news_horizon, effect_horizon, max_quotes_lag, show_shapes=False, news_show=False):
    d = './data/data/rex.xlsx'
    data = pandas.read_excel(d)

    newstitle_frame = data[['id', 'time', 'title']]
    lag_markers = list(itertools.product(newstitle_frame['id'].values, numpy.array(numpy.arange(news_horizon - 1)) + 1))
    lag_markers = pandas.DataFrame(data=lag_markers, columns=['id', 'lag'])
    newstitle_frame = newstitle_frame.merge(right=lag_markers, left_on=['id'], right_on=['id'])

    newstitle_frame['target_date'] = newstitle_frame.apply(func=add_lag, axis=1)
    beginning_date, ending_date = newstitle_frame['target_date'].min() - datetime.timedelta(
        days=(max_quotes_lag + effect_horizon)), newstitle_frame['target_date'].max()

    beginning_date = datetime.datetime.combine(beginning_date, datetime.datetime.min.time())
    ending_date = datetime.datetime.combine(ending_date, datetime.datetime.min.time())

    quotes_data = await call_them_all(tickers=target_quotes,
                                      start_date=beginning_date, end_date=ending_date,
                                      token=api_key)
    quotes_data = quotes_data.set_index(keys=['ticker', 'time'])
    quotes_data = quotes_data.sort_index(ascending=True)

    quotes_data = fill_all(frame=quotes_data, freq='T', zero_index_name='ticker', first_index_name='time')

    quotes_data = consequentive_lagger(frame=quotes_data, n_lags=effect_horizon, suffix='_HOZ')

    quotes_data = consequentive_pcter(frame=quotes_data, horizon=1)

    quotes_data = quotes_data.reset_index()
    quotes_data['time'] = quotes_data['time'].apply(func=to_date)
    
    print(quotes_data['time'])
    print(newstitle_frame['target_date'])
    
    the_data = quotes_data.merge(right=newstitle_frame, left_on='time', right_on='target_date')
    print(newstitle_frame['title'].value_counts().shape)

    return the_data


def add_lag(x):
    a = x['time'] + pandas.DateOffset(days=x['lag'])
    return datetime.date(a.year, a.month, a.day)


def lagger(frame, n_lags):
    frame_ = frame.copy()
    if frame_.index.nlevels == 1:
        frame_ = frame_.shift(periods=n_lags, axis=0)
    elif frame_.index.nlevels == 2:
        for ix in frame_.index.levels[0]:
            frame_.loc[[ix], :] = frame_.loc[[ix], :].shift(periods=n_lags, axis=0)
    else:
        raise NotImplemented()
    return frame_


def consequentive_lagger(frame, n_lags, exactly=True, keep_basic=True, suffix='_LAG'):
    if exactly:
        if keep_basic:
            new_columns = [x + suffix + '0' for x in frame.columns.values] + [x + suffix + str(n_lags) for x in frame.columns.values]
            frame = pandas.concat((frame, lagger(frame=frame, n_lags=n_lags)), axis=1)
            frame.columns = new_columns
        else:
            new_columns = [x + suffix + str(n_lags) for x in frame.columns.values]
            frame = lagger(frame=frame, n_lags=n_lags)
            frame.columns = new_columns
    else:
        if keep_basic:
            new_columns = [x + suffix + '0' for x in frame.columns.values]
            frames = [frame]
        else:
            new_columns = []
            frames = []
        for j in numpy.arange(start=1, stop=(n_lags + 1)):
            new_columns = new_columns + [x + suffix + str(j) for x in frame.columns.values]
            frames.append(lagger(frame=frame, n_lags=j))
        frame = pandas.concat(frames, axis=1)
        frame.columns = new_columns
    return frame


def pcter(frame, n_lags):
    frame_ = frame.copy()
    if frame_.index.nlevels == 1:
        frame_ = frame_.pct_change(periods=n_lags, axis=0)
    elif frame_.index.nlevels == 2:
        for ix in frame_.index.levels[0]:
            frame_.loc[[ix], :] = frame_.loc[[ix], :].pct_change(periods=n_lags, axis=0)
    else:
        raise NotImplemented()
    return frame_


def consequentive_pcter(frame, horizon, exactly=True, suffix='_PCT'):
    if exactly:
        new_columns = [x + suffix + str(horizon) for x in frame.columns.values]
        frame = pcter(frame=frame, n_lags=horizon)
        frame.columns = new_columns
    else:
        new_columns = []
        frames = []
        for j in numpy.arange(start=1, stop=(horizon + 1)):
            new_columns = new_columns + [x + suffix + str(j) for x in frame.columns.values]
            frames.append(pcter(frame=frame, n_lags=j))
        frame = pandas.concat(frames, axis=1)
        frame.columns = new_columns
    return frame


def lag_old(array, names, ex=None, upon=None, exactly=None, appx='LAG'):
    if ex is None:
        ex = []
    if upon is not None:
        result, rname = [], []
        arra = array[:, [x not in ex for x in names]]
        for j in range((upon + 1)):
            re = numpy.roll(arra, shift=j, axis=0)
            re[:j] = numpy.nan
            result.append(re)
            rname.append([x + '_{}{}'.format(appx, j) for x in names if x not in ex])
        return numpy.concatenate((numpy.concatenate(result, axis=1), array[:, [x in ex for x in names]]),
                                 axis=1), numpy.concatenate((numpy.concatenate(rname), ex))
    if exactly is not None:
        result, rname = [], []
        arra = array[:, [x not in ex for x in names]]
        for j in [0, exactly]:
            re = numpy.roll(arra, shift=j, axis=0)
            re[:j] = numpy.nan
            result.append(re)
            rname.append([x + '_{}{}'.format(appx, j) for x in names if x not in ex])
        return numpy.concatenate((numpy.concatenate(result, axis=1), array[:, [x in ex for x in names]]),
                                 axis=1), numpy.concatenate((numpy.concatenate(rname), ex))


def to_date_old(x):
    return datetime.date(int(x[:4]), int(x[5:7]), int(x[8:]))


def to_date(x):
    return x.date()


def filler(frame, date_start, date_end, freq, tz):
    result = pandas.DataFrame(index=pandas.date_range(start=date_start, end=date_end, freq=freq, tz=tz),
                              data=frame)
    return result


def fill_all(frame, freq, zero_index_name, first_index_name):
    data = []
    for ix0 in frame.index.levels[0]:
        filled = filler(frame=frame.loc[ix0, :], date_start=frame.index.levels[1].min(), date_end=frame.index.levels[1].max(), freq=freq, tz=frame.index.levels[1][0].tz)
        filled = filled.reset_index()
        filled[zero_index_name] = ix0
        filled = filled.rename(columns={'index': first_index_name})
        data.append(filled)
    data = pandas.concat(data, axis=0)
    data = data.set_index(keys=[zero_index_name, first_index_name])
    return data

