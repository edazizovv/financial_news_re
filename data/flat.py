import pandas
import datetime
import pandas_datareader
import itertools
import numpy

"""
IDK what's this, they just came from somewhere...

for:   dayXnewstitle, ticker, news_horizont (how much it lasted from the news' release), effect_horizont (the horizon we consider effect on) 
spec (most general):  affected_target ~ (self lags), (related targets from other observations), (related objects), (indicators), (macro)

dimension0 (kinds): yield itself, yield variance, specific properties of it's process

dimension1 (regime):
    group
        model -- by group
        (groups: stocks, bonds, etfs, indicators, goods)
    individual
        model -- by quote

dimension2 (variety of fields):
    1 = self only
    2 = 1 + others from its config observ
    3 = 1 + news
    4 = 1 + others from its config observ + news

dimension 3 (time axis specification):
    no specification
    simple ordering (1, 2, 3 ...)
    day parameters (timestamp components):
        year
        season (summer, winter etc.)
        month
        week
        day
        part of day
        hour
        minute
        second
    general economy activity parameters: 
        business day or not
        business session or not
        financial year
    trading activity parameters: 
        trading day or not
        session is opened or not
    relational -- timestamp components:
        until/since next/last:
            business day
            business session
            trading day
            trading session
            weekend
            end of year
            end of financial year
            
            ! special market events !
        
"""


def load(api_key, target_quotes, news_horizon, effect_horizon, max_quotes_lag):

    d = './data/data/rex.xlsx'
    data = pandas.read_excel(d)

    newstitle_frame = data[['id', 'time', 'title']]
    lag_markers = list(itertools.product(newstitle_frame['id'].values, numpy.array(numpy.arange(news_horizon - 1)) + 1))
    lag_markers = pandas.DataFrame(data=lag_markers, columns=['id', 'lag'])
    newstitle_frame = newstitle_frame.merge(right=lag_markers, left_on=['id'], right_on=['id'])

    newstitle_frame['target_date'] = newstitle_frame.apply(func=add_lag, axis=1)
    beginning_date, ending_date = newstitle_frame['target_date'].min() - datetime.timedelta(days=(max_quotes_lag + effect_horizon)), newstitle_frame['target_date'].max()

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

    quotes_data_lagged_values, quotes_data_lagged_columns = lag(array=quotes_data.values,
                                                                names=quotes_data.columns.values,
                                                                exactly=effect_horizon, appx='hori', ex=['ticker'])
    quotes_data = pandas.DataFrame(data=quotes_data_lagged_values, index=quotes_data.index.values,
                                   columns=quotes_data_lagged_columns)

    for j in range(nn):
        quotes_data.iloc[:, 5 + j] = quotes_data.iloc[:, 5 + j] / quotes_data.iloc[:, j] - 1
    quotes_data = quotes_data.drop(columns=[quotes_data.columns.values[j] for j in range(nn)])

    quotes_data = quotes_data.dropna()

    quotes_data_lagged_values, quotes_data_lagged_columns = lag(array=quotes_data.values,
                                                                names=quotes_data.columns.values, upon=max_quotes_lag,
                                                                ex=['ticker'])
    quotes_data_lagged = pandas.DataFrame(data=quotes_data_lagged_values, index=quotes_data.index.values,
                                          columns=quotes_data_lagged_columns)

    quotes_data_lagged = quotes_data_lagged.dropna()
    quotes_data_lagged = quotes_data_lagged.reset_index()
    quotes_data_lagged['index'] = quotes_data_lagged['index'].apply(func=to_date)
    the_data = quotes_data_lagged.merge(right=newstitle_frame, left_on='index', right_on='target_date')

    return the_data


def add_lag(x):
    a = x['time'] + pandas.DateOffset(days=x['lag'])
    return datetime.date(a.year, a.month, a.day)


def lag(array, names, ex, upon=None, exactly=None, appx='LAG'):
    if upon is not None:
        result, rname = [], []
        arra = array[:, [x not in ex for x in names]]
        for j in range((upon + 1)):
            re = numpy.roll(arra, shift=j, axis=0)
            re[:j] = numpy.nan
            result.append(re)
            rname.append([x + '_{}{}'.format(appx, j) for x in names if x not in ex])
        return numpy.concatenate((numpy.concatenate(result, axis=1), array[:, [x in ex for x in names]]), axis=1), numpy.concatenate((numpy.concatenate(rname), ex))
    if exactly is not None:
        result, rname = [], []
        arra = array[:, [x not in ex for x in names]]
        for j in [0, exactly]:
            re = numpy.roll(arra, shift=j, axis=0)
            re[:j] = numpy.nan
            result.append(re)
            rname.append([x + '_{}{}'.format(appx, j) for x in names if x not in ex])
        return numpy.concatenate((numpy.concatenate(result, axis=1), array[:, [x in ex for x in names]]), axis=1), numpy.concatenate((numpy.concatenate(rname), ex))


def to_date(x):
    return datetime.date(int(x[:4]), int(x[5:7]), int(x[8:]))

