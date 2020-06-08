import pandas
import pandas_datareader
import datetime
import itertools
import numpy
import json
import sqlalchemy


# https://github.com/Fatal1ty/tinkoff-api


class Loader:

    def __init__(self, api_key, target_quotes, news_horizon, effect_horizon, db_config, reload_quotes=False,
                 news_titles_source=None, verbose=False):

        self.verbose = verbose

        self.api_key = api_key
        self.target_quotes = target_quotes
        self.news_horizon = news_horizon
        self.effect_horizon = effect_horizon
        self.db_config = db_config
        self.reload_quotes = reload_quotes
        self.news_titles_source = news_titles_source
        self.connection = None
        self.news_titles_frame = None
        self.quotes_frame = None

        self.news_titles_raw_alias = 'news_titles_raw'
        self.news_titles_alias = 'news_titles'
        self.quotes_alias = 'quotes'
        self.mid_name = 'mid_table'

    def establish_connection(self):

        if self.verbose:
            print('Establishing Connection')

        with open(self.db_config) as f:
            db_config = json.load(f)

        user, password, host, port, dbname = db_config['user'], db_config['password'], db_config['host'], db_config[
            'port'], db_config['dbname']

        connection_string = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(user, password, host, port, dbname)
        engine = sqlalchemy.create_engine(connection_string)

        self.connection = engine.connect()

    def prepare_news_titles_frame(self):

        if self.verbose:
            print('Preparing News Titles Frame')

        if self.news_titles_source is not None:
            self.news_titles_frame = pandas.read_excel(self.news_titles_source)

            self.news_titles_frame['time'] = pandas.to_datetime(self.news_titles_frame['time'])

            def fix_tz(x):
                return x.tz_localize(tz='UTC')

            self.news_titles_frame['time'] = self.news_titles_frame['time'].apply(func=fix_tz)

            def fixit(x):
                return x.ceil(freq='T')

            self.news_titles_frame['time'] = self.news_titles_frame['time'].apply(func=fixit)

            self.news_titles_frame.to_sql(name=self.news_titles_raw_alias, con=self.connection, if_exists='replace',
                                          index=False)

            # to sql !

            news_titles_final_query = """
            CREATE TEMPORARY TABLE {2} AS
            
                WITH cutta AS
                (
                SELECT id, time, title
                FROM {0}
                ),
                listed AS
                (
                SELECT generate_series(1, {1}) AS lags
                ),
                identified AS
                (
                SELECT DISTINCT id
                FROM {0}
                ),
                crossy AS
                (
                SELECT identified.id, listed.lags
                FROM
                identified
                CROSS JOIN
                listed
                )
                SELECT cutta.id, cutta.time AS news_time, title, (cutta.time + (crossy.lags * INTERVAL '1 minute')) AS time
                FROM
                cutta
                INNER JOIN
                crossy
                ON cutta.id = crossy.id
            ;
            """.format(self.news_titles_raw_alias, self.news_horizon, self.news_titles_alias)

            self.connection.execute(news_titles_final_query)

    def get_dates(self):

        if self.verbose:
            print('Getting Dates')

        beginning_date_query = """
        SELECT (MIN(time) - ({1} * INTERVAL '1 minute')) AS mn
        FROM {0}
        """.format(self.news_titles_raw_alias, self.effect_horizon)

        ending_date_query = """
        SELECT MAX(time) as mx
        FROM {0}
        """.format(self.news_titles_raw_alias)

        beginning_date = pandas.read_sql(sql=beginning_date_query, con=self.connection).values[0, 0]
        ending_date = pandas.read_sql(sql=ending_date_query, con=self.connection).values[0, 0]

        return beginning_date, ending_date

    async def call_quotes(self):

        beginning_date, ending_date = self.get_dates()

        self.quotes_frame = await call_them_all(tickers=self.target_quotes, start_date=beginning_date,
                                                end_date=ending_date, token=self.api_key)

    async def prepare_quotes(self):

        if self.verbose:
            print('Preparing Quotes')

        if self.reload_quotes:

            # loop = asyncio.get_event_loop()
            # loop.run_until_complete(self.call_quotes())
            # loop.close()

            await self.call_quotes()

            self.quotes_frame.to_sql(name=self.quotes_alias, con=self.connection, if_exists='replace', index=False)

    def quotes_fill(self):

        if self.verbose:
            print('Filling Quotes')

        beginning_date, ending_date = self.get_dates()

        pre_query = """
        CREATE TEMPORARY TABLE temp_table AS 
            SELECT generate_series(TIMESTAMP WITH TIME ZONE '{0}', TIMESTAMP WITH TIME ZONE '{1}', '1 minute') AS "time"
        ;
        """.format(beginning_date, ending_date)
        self.connection.execute(pre_query)

        quotes_data_columns_query = """
        SELECT *
        FROM {0}
        LIMIT 1
        ;
        """.format(self.quotes_alias)

        quotes_columns = pandas.read_sql(sql=quotes_data_columns_query, con=self.connection).columns.values
        cc = [self.quotes_alias + '.' + x for x in quotes_columns if x not in ['time', 'ticker']]

        mid_query = """
        CREATE TEMPORARY TABLE {2} AS

            SELECT src.time, src.ticker, {1}
            FROM 
                 {0}

                 RIGHT JOIN 

                 (SELECT *
                 FROM

                 temp_table

                 CROSS JOIN

                 (SELECT DISTINCT ticker
                 FROM {0}) AS krol
                 ) AS src


                 ON {0}.time = src.time AND {0}.ticker = src.ticker
        ;
        """.format(self.quotes_alias, forma(cc), self.mid_name)
        self.connection.execute(mid_query)

    def quotes_lag(self):

        if self.verbose:
            print('Lagging Quotes')

        get_cols = """
        SELECT *
        FROM {0}
        LIMIT 1
        ;
        """.format(self.mid_name)

        identifiers = ['ticker']
        identifiers_and_time = ['time'] + identifiers
        lagging_columns = [x for x in pandas.read_sql(sql=get_cols, con=self.connection).columns.values if
                           x not in identifiers_and_time]

        lag_alias = 'LAG'
        n_lags = self.effect_horizon
        for column in lagging_columns:
            query_execute = """
            -- https://stackoverflow.com/questions/13289304/postgresql-dynamic-value-as-table-name
            DO
            $$
            DECLARE t_name TEXT;
            DECLARE c_name TEXT;
            DECLARE c_type TEXT;
            DECLARE l_alias TEXT;
            DECLARE n_lags INT;
            BEGIN
            t_name = '{0}';
            c_name = '{1}';
            l_alias = '{2}';
            n_lags = {3};
            SELECT data_type FROM information_schema.columns
            WHERE 37=37
            AND table_name = t_name
            AND column_name = c_name
            INTO c_type;
            EXECUTE format('
                           ALTER TABLE %%I
                           ADD COLUMN tmp_id TEXT;
                           UPDATE %%I
                           SET tmp_id = CONCAT({5});',
                           t_name,
                           t_name);
            FOR i IN 1..n_lags LOOP
                EXECUTE format('
                               ALTER TABLE %%I
                               ADD COLUMN %%I %%s ;
                               WITH new_cc AS
                               (
                               SELECT tmp_id, LAG(%%I, %%s) OVER (PARTITION BY {4}) AS lg
                               FROM %%I
                               )
                               UPDATE %%I
                               SET %%I = new_cc.lg
                               FROM new_cc
                               WHERE %%I.tmp_id = new_cc.tmp_id', 
                               t_name,
                               c_name || '_' || l_alias || i, 
                               c_type,
                               c_name, 
                               i,
                               t_name,
                               t_name,
                               c_name || '_LAG' || i,
                               t_name);
            END LOOP;
            EXECUTE format('
                           ALTER TABLE %%I
                           DROP COLUMN tmp_id; ',
                           t_name);
            END;
            $$ LANGUAGE plpgsql;
            """.format(self.mid_name, column, lag_alias, n_lags, forma(identifiers), forma(identifiers_and_time))
            self.connection.execute(query_execute)

    def quotes_percent(self):

        if self.verbose:
            print('Evaluating Quotes Percents')

        get_cols = """
        SELECT *
        FROM {0}
        LIMIT 1
        """.format(self.mid_name)

        identifiers = ['ticker']
        identifiers_and_time = ['time'] + identifiers
        columns = [x for x in pandas.read_sql(sql=get_cols, con=self.connection).columns.values if
                   x not in identifiers_and_time]

        for column in columns:
            query_execute = """
            DO
            $$
            DECLARE t_name TEXT;
            DECLARE c_name TEXT;
            BEGIN
            t_name = '{0}';
            c_name = '{1}';
            EXECUTE format('
                           ALTER TABLE %%I
                           ADD COLUMN tmp_id TEXT;
                           UPDATE %%I
                           SET tmp_id = CONCAT({3});',
                           t_name,
                           t_name);
            EXECUTE format('
                           WITH new_cc AS
                           (
                           SELECT tmp_id, (%%I / LAG(%%I, 1) OVER (PARTITION BY {2}) - 1) AS pc
                           FROM %%I
                           )
                           UPDATE %%I
                           SET %%I = new_cc.pc
                           FROM new_cc
                           WHERE %%I.tmp_id = new_cc.tmp_id', 
                           c_name, 
                           c_name, 
                           t_name,
                           t_name,  
                           c_name,
                           t_name);
            EXECUTE format('
                           ALTER TABLE %%I
                           DROP COLUMN tmp_id; ',
                           t_name);
            END;
            $$ LANGUAGE plpgsql;
            """.format(self.mid_name, column, forma(identifiers), forma(identifiers_and_time))

            self.connection.execute(query_execute)

    async def read(self):

        if self.verbose:
            print('Reading')

        self.establish_connection()
        self.prepare_news_titles_frame()
        await self.prepare_quotes()
        self.quotes_fill()
        self.quotes_lag()
        self.quotes_percent()

        query = """
         SELECT RS.*
         FROM
            (SELECT NF."id"
                  , NF.title
                  , NF."lag"
                  , NF.news_time
                  , QD.*
            FROM
            public.newstitle_frame AS NF
            FULL OUTER JOIN
            public.{0} AS QD
            ON NF."time" = QD."time") AS RS
         WHERE 37 = 37
         ;
         """.format(self.mid_name)

        the_data = pandas.read_sql(sql=query, con=self.connection)
        return the_data


def forma(x):
    return str(x).replace('[', '').replace(']', '').replace("'", '')


import asyncio
from tinkoff.investments import (
    TinkoffInvestmentsRESTClient, Environment, CandleResolution
)
from tinkoff.investments.client.exceptions import TinkoffInvestmentsError


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


async def load(api_key, target_quotes, news_horizon, effect_horizon, max_quotes_lag=None, show_shapes=False,
               news_show=False):
    d = './data/data/rex.xlsx'
    data = pandas.read_excel(d)

    newstitle_frame = data[['id', 'time', 'title']]
    lag_markers = list(itertools.product(newstitle_frame['id'].values, numpy.array(numpy.arange(news_horizon - 1)) + 1))
    lag_markers = pandas.DataFrame(data=lag_markers, columns=['id', 'lag'])
    newstitle_frame = newstitle_frame.merge(right=lag_markers, left_on=['id'], right_on=['id'])

    newstitle_frame['time'] = pandas.to_datetime(newstitle_frame['time'])
    newstitle_frame['news_time'] = newstitle_frame['time'].copy()
    # newstitle_frame['time'] = newstitle_frame.apply(func=add_lag, axis=1)
    newstitle_frame['time'] = newstitle_frame['lag'].apply(func=minute_offset)
    newstitle_frame['time'] = newstitle_frame['news_time'] + newstitle_frame['time']
    beginning_date, ending_date = newstitle_frame['time'].min() - pandas.DateOffset(
        minutes=effect_horizon), newstitle_frame['time'].max()

    beginning_date = datetime.datetime.combine(beginning_date, datetime.datetime.min.time())
    ending_date = datetime.datetime.combine(ending_date, datetime.datetime.min.time())

    print(beginning_date)
    print(ending_date)

    quotes_data = await call_them_all(tickers=target_quotes,
                                      start_date=beginning_date, end_date=ending_date,
                                      token=api_key)

    # --------------

    with open('E:/InverseStation/terminator_panel/users.json') as f:
        users = json.load(f)

    user, password = users['justiciar']['user'], users['justiciar']['password']

    with open('E:/InverseStation/terminator_panel/servers.json') as f:
        users = json.load(f)

    host, port = users['GOLA']['host'], users['GOLA']['port']

    dbname = 'tempbox'

    connection_string = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(user, password, host, port, dbname)
    engine = sqlalchemy.create_engine(connection_string)
    connection = engine.connect()

    # filler
    temp_name = 'temp_tbl'
    mid_name = 'mid_table'

    quotes_data.to_sql(name=temp_name, con=connection, if_exists='replace', index=False)

    pre_query = """
    CREATE TEMPORARY TABLE temp_table AS 
        SELECT generate_series(TIMESTAMP WITH TIME ZONE '{0}', TIMESTAMP WITH TIME ZONE '{1}', '1 minute') AS "time"
    ;
    """.format(beginning_date, ending_date)
    connection.execute(pre_query)
    cc = [temp_name + '.' + x for x in quotes_data.columns if x not in ['time', 'ticker']]

    mid_query = """
    CREATE TEMPORARY TABLE {2} AS
    
        SELECT src.time, src.ticker, {1}
        FROM 
             {0}
             
             RIGHT JOIN 
             
             (SELECT *
             FROM
             
             temp_table
             
             CROSS JOIN
             
             (SELECT DISTINCT ticker
             FROM {0}) AS krol
             ) AS src
             
             
             ON {0}.time = src.time AND {0}.ticker = src.ticker
    ;
    """.format(temp_name, forma(cc), mid_name)
    # print(mid_query)
    # quotes_data = pandas.read_sql(sql=mid_query, con=connection)
    connection.execute(mid_query)

    # lagger

    # data.to_sql(name=temp_name, con=conn, if_exists='replace', index=False)

    get_cols = """
    SELECT *
    FROM {0}
    LIMIT 1
    ;
    """.format(mid_name)
    identifiers = ['ticker']
    identifiers_and_time = ['time'] + identifiers
    lagging_columns = [x for x in pandas.read_sql(sql=get_cols, con=connection).columns.values if
                       x not in identifiers_and_time]
    # print(lagging_columns)
    lag_alias = 'LAG'
    n_lags = effect_horizon
    for column in lagging_columns:
        query_execute = """
        -- https://stackoverflow.com/questions/13289304/postgresql-dynamic-value-as-table-name
        DO
        $$
        DECLARE t_name TEXT;
        DECLARE c_name TEXT;
        DECLARE c_type TEXT;
        DECLARE l_alias TEXT;
        DECLARE n_lags INT;
        BEGIN
        t_name = '{0}';
        c_name = '{1}';
        l_alias = '{2}';
        n_lags = {3};
        SELECT data_type FROM information_schema.columns
        WHERE 37=37
        AND table_name = t_name
        AND column_name = c_name
        INTO c_type;
        EXECUTE format('
                       ALTER TABLE %%I
                       ADD COLUMN tmp_id TEXT;
                       UPDATE %%I
                       SET tmp_id = CONCAT({5});',
                       t_name,
                       t_name);
        FOR i IN 1..n_lags LOOP
            EXECUTE format('
                           ALTER TABLE %%I
                           ADD COLUMN %%I %%s ;
                           WITH new_cc AS
                           (
                           SELECT tmp_id, LAG(%%I, %%s) OVER (PARTITION BY {4}) AS lg
                           FROM %%I
                           )
                           UPDATE %%I
                           SET %%I = new_cc.lg
                           FROM new_cc
                           WHERE %%I.tmp_id = new_cc.tmp_id', 
                           t_name,
                           c_name || '_' || l_alias || i, 
                           c_type,
                           c_name, 
                           i,
                           t_name,
                           t_name,
                           c_name || '_LAG' || i,
                           t_name);
        END LOOP;
        EXECUTE format('
                       ALTER TABLE %%I
                       DROP COLUMN tmp_id; ',
                       t_name);
        END;
        $$ LANGUAGE plpgsql;
        """.format(mid_name, column, lag_alias, n_lags, forma(identifiers), forma(identifiers_and_time))
        # .format(temp_name, column, lag_alias, n_lags, forma(identifiers), forma(identifiers_and_time))
        # print(query_execute)
        connection.execute(query_execute)

    query_select = """
    SELECT *
    FROM {0}
    ;
    """.format(mid_name)
    # .format(temp_name)
    # data = pandas.read_sql(sql=query_select, con=connection)
    # print(data)
    # raise Exception("Hands UP!")

    # pcter

    get_cols = """
    SELECT *
    FROM {0}
    LIMIT 1
    """.format(mid_name)

    identifiers = ['ticker']
    identifiers_and_time = ['time'] + identifiers
    columns = [x for x in pandas.read_sql(sql=get_cols, con=connection).columns.values if x not in identifiers_and_time]
    print(columns)
    for column in columns:
        query_execute = """
        DO
        $$
        DECLARE t_name TEXT;
        DECLARE c_name TEXT;
        BEGIN
        t_name = '{0}';
        c_name = '{1}';
        EXECUTE format('
                       ALTER TABLE %%I
                       ADD COLUMN tmp_id TEXT;
                       UPDATE %%I
                       SET tmp_id = CONCAT({3});',
                       t_name,
                       t_name);
        EXECUTE format('
                       WITH new_cc AS
                       (
                       SELECT tmp_id, (%%I / LAG(%%I, 1) OVER (PARTITION BY {2}) - 1) AS pc
                       FROM %%I
                       )
                       UPDATE %%I
                       SET %%I = new_cc.pc
                       FROM new_cc
                       WHERE %%I.tmp_id = new_cc.tmp_id', 
                       c_name, 
                       c_name, 
                       t_name,
                       t_name,  
                       c_name,
                       t_name);
        EXECUTE format('
                       ALTER TABLE %%I
                       DROP COLUMN tmp_id; ',
                       t_name);
        END;
        $$ LANGUAGE plpgsql;
        """.format(mid_name, column, forma(identifiers), forma(identifiers_and_time))
        print(query_execute)
        connection.execute(query_execute)

    read_quotes = """
    SELECT *
    FROM {0}
    ;
    """.format(mid_name)

    quotes_data = pandas.read_sql(sql=read_quotes, con=connection)

    print(quotes_data)
    # quotes_data = quotes_data.set_index(keys=['ticker', 'time'])
    # quotes_data = quotes_data.sort_index(ascending=True)
    '''
    quotes_data = fill_all(frame=quotes_data, freq='T', zero_index_name='ticker', first_index_name='time')
    
    quotes_data = consequentive_lagger(frame=quotes_data, n_lags=effect_horizon, suffix='_HOZ', exactly=False)
    
    quotes_data = consequentive_pcter(frame=quotes_data, horizon=1)

    quotes_data = quotes_data.reset_index()

    print(quotes_data.shape)
    print(quotes_data.columns)
    print(quotes_data)
    '''
    # quotes_data['time'] = quotes_data['time'].apply(func=to_date)
    print(1)

    newstitle_frame['time'] = pandas.to_datetime(newstitle_frame['time'])
    # quotes_data['time'] = pandas.to_datetime(quotes_data['time'])

    qd_tz = quotes_data.loc[0, 'time'].tz

    def fix_tz(x):
        return x.tz_localize(tz=qd_tz)

    newstitle_frame['time'] = newstitle_frame['time'].apply(func=fix_tz)

    def fixit(x):
        return x.ceil(freq='T')

    # quotes_data['time'] = quotes_data['time'].apply(func=fixit)
    newstitle_frame['time'] = newstitle_frame['time'].apply(func=fixit)
    print(2)

    # quotes_data.to_sql(name='quotes_data', con=connection, if_exists='replace', index=False)
    newstitle_frame.to_sql(name='newstitle_frame', con=connection, if_exists='replace', index=False)

    query = """
    SELECT RS.*
    FROM
    	(SELECT NF."id"
    		 , NF.title
    		 , NF."lag"
    		 , NF.news_time
    		 , QD.*
    	FROM
    	public.newstitle_frame AS NF
    	FULL OUTER JOIN
    	public.{0} AS QD
    	ON NF."time" = QD."time") AS RS
    WHERE 37 = 37
    ;
    """.format(mid_name)
    print(3)
    the_data = pandas.read_sql(sql=query, con=connection)

    # print(quotes_data['time'])
    # print(newstitle_frame['target_date'])
    #
    # the_data = quotes_data.merge(right=newstitle_frame, left_on='time', right_on='target_date')
    print(newstitle_frame['title'].value_counts().shape)
    print(4)
    return the_data


def minute_offset(x):
    return pandas.DateOffset(minutes=x)


def add_lag(x):
    # a = x['time'] + pandas.DateOffset(days=x['lag'])
    # a = x['time'] + pandas.DateOffset(minutes=x['lag'])
    # return datetime.date(a.year, a.month, a.day)
    return x['time'] + pandas.DateOffset(minutes=x['lag'])


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
            new_columns = [x + suffix + '0' for x in frame.columns.values] + [x + suffix + str(n_lags) for x in
                                                                              frame.columns.values]
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
        frame_ = frame_.pct_change(periods=n_lags, axis=0, fill_method=None)
    elif frame_.index.nlevels == 2:
        for ix in frame_.index.levels[0]:
            frame_.loc[[ix], :] = frame_.loc[[ix], :].pct_change(periods=n_lags, axis=0, fill_method=None)
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
        filled = filler(frame=frame.loc[ix0, :], date_start=frame.index.levels[1].min(),
                        date_end=frame.index.levels[1].max(), freq=freq, tz=frame.index.levels[1][0].tz)
        filled = filled.reset_index()
        filled[zero_index_name] = ix0
        filled = filled.rename(columns={'index': first_index_name})
        data.append(filled)
    data = pandas.concat(data, axis=0)
    data = data.set_index(keys=[zero_index_name, first_index_name])
    return data
