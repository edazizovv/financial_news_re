#
import os
import time
import pandas
import itertools
import numpy
import json
import sqlalchemy

from tinkoff_api.quotes_loader import call_them_all
from m_utils.transform import lag_it, percent_it, fill_it


def sql_formatting(x):
    return str(x).replace('[', '').replace(']', '').replace("'", '')


class SparseLoader:

    def __init__(self, api_key, target_quotes, news_horizon, effect_horizon, db_config, reload_quotes=False,
                 news_titles_source=None, verbose=False, timeit=False, base_option='for_merge', add_time_features=False,
                 nlp_treator=None, nlp_treator_signature=None, nlp_treator_config=None, nlp_ductor='post',
                 export_chunk=100_000):

        self.verbose = verbose
        self.timeit = timeit
        self.run_time = None
        self.base_option = base_option
        self.add_time_features = add_time_features

        self.export_chunk = export_chunk
        self.where_to_save = './result.csv'

        self.nlp_treator = nlp_treator
        self.nlp_treator_signature = nlp_treator_signature
        self.nlp_treator_config = nlp_treator_config
        self.nlp_ductor = nlp_ductor

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

        self.news_titles_alias = 'news_titles'
        self.quotes_alias = 'quotes'
        self.result_alias = 'result_table'

    def fix_time(self):

        self.run_time = time.time()

    def do_time(self):

        self.run_time = time.time() - self.run_time
        print(self.run_time)

    def establish_connection(self):

        if self.timeit:
            self.fix_time()

        if self.base_option == 'for_merge':

            if self.verbose:
                print('Establishing Connection')

            with open(self.db_config) as f:
                db_config = json.load(f)

            user, password, host, port, dbname = db_config['user'], db_config['password'], db_config['host'], db_config[
                'port'], db_config['dbname']

            connection_string = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(user, password, host, port, dbname)
            engine = sqlalchemy.create_engine(connection_string)

            self.connection = engine.connect()

            if self.timeit:
                self.do_time()

        else:

            if self.verbose:
                print('Skipped Connection')

    def prepare_news_titles_frame(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Preparing News Titles Frame')

        if self.news_titles_source is None:
            raise Exception("You should specify news titles source")

        if self.news_titles_source is not None:

            self.news_titles_frame = pandas.read_excel(self.news_titles_source)

            self.news_titles_frame['time'] = pandas.to_datetime(self.news_titles_frame['time'])

            def fix_tz(x):
                return x.tz_localize(tz='UTC')

            self.news_titles_frame['time'] = self.news_titles_frame['time'].apply(func=fix_tz)

            def fixit(x):
                return x.ceil(freq='T')

            self.news_titles_frame['time'] = self.news_titles_frame['time'].apply(func=fixit)

            if self.nlp_treator is not None and self.nlp_ductor == 'pre':
                old_name = 'title'
                new_name = 'Text'
                self.news_titles_frame = self.news_titles_frame.rename(columns={old_name: new_name})
                self.news_titles_frame = self.nlp_treator(self.news_titles_frame,
                                                          self.nlp_treator_signature, self.nlp_treator_config)

                self.news_titles_frame = self.nlp_treator(self.news_titles_frame,
                                                          self.nlp_treator_signature, self.nlp_treator_config)

                self.news_titles_frame = self.news_titles_frame.rename(columns={new_name: old_name})

            # self.news_titles_frame = self.news_titles_frame[['id', 'time', 'title']]
            self.news_titles_frame = self.news_titles_frame.drop(columns=['source', 'category'])
            lag_markers = list(
                itertools.product(self.news_titles_frame['id'].values,
                                  numpy.array(numpy.arange(self.news_horizon - 1)) + 1))
            lag_markers = pandas.DataFrame(data=lag_markers, columns=['id', 'lag'])
            self.news_titles_frame = self.news_titles_frame.merge(right=lag_markers, left_on=['id'],
                                                                  right_on=['id'])

            def minute_offset(x):
                return pandas.DateOffset(minutes=x)

            self.news_titles_frame['time'] = pandas.to_datetime(self.news_titles_frame['time'])
            self.news_titles_frame['news_time'] = self.news_titles_frame['time'].copy()
            self.news_titles_frame['time'] = self.news_titles_frame['lag'].apply(func=minute_offset)
            self.news_titles_frame['time'] = self.news_titles_frame['news_time'] + self.news_titles_frame['time']

            if self.base_option == 'for_merge':
                self.news_titles_frame.to_sql(name=self.news_titles_alias, con=self.connection,
                                              if_exists='replace',
                                              index=False)

        if self.timeit:
            self.do_time()

    def get_dates(self):

        if self.verbose:
            print('Getting Dates')

        if self.base_option == 'without':

            beginning_date, ending_date = self.news_titles_frame['time'].min() - pandas.DateOffset(
                minutes=self.effect_horizon), self.news_titles_frame['time'].max()

        else:

            beginning_date_query = """
            SELECT (MIN(time) - ({1} * INTERVAL '1 minute')) AS mn
            FROM {0}
            """.format(self.news_titles_alias, self.effect_horizon)

            ending_date_query = """
            SELECT MAX(time) as mx
            FROM {0}
            """.format(self.news_titles_alias)

            beginning_date = pandas.read_sql(sql=beginning_date_query, con=self.connection).values[0, 0]
            ending_date = pandas.read_sql(sql=ending_date_query, con=self.connection).values[0, 0]

        return beginning_date, ending_date

    async def call_quotes(self):

        beginning_date, ending_date = self.get_dates()

        self.quotes_frame = await call_them_all(tickers=self.target_quotes, start_date=beginning_date,
                                                end_date=ending_date, token=self.api_key)

    async def prepare_quotes(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Preparing Quotes')

        if self.reload_quotes:

            await self.call_quotes()

        if self.timeit:
            self.do_time()

    def quotes_fill(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Filling Quotes')

        beginning_date, ending_date = self.get_dates()

        self.quotes_frame = self.quotes_frame.set_index(keys=['ticker', 'time'])
        self.quotes_frame = self.quotes_frame.sort_index(ascending=True)
        self.quotes_frame = fill_it(frame=self.quotes_frame, freq='T', zero_index_name='ticker',
                                    first_index_name='time')
        self.quotes_frame = self.quotes_frame.reset_index()

        if self.timeit:
            self.do_time()

    def quotes_lag(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Lagging Quotes')

        self.quotes_frame = self.quotes_frame.set_index(keys=['ticker', 'time'])
        self.quotes_frame = self.quotes_frame.sort_index(ascending=True)
        self.quotes_frame = lag_it(frame=self.quotes_frame, n_lags=self.effect_horizon, suffix='_LAG',
                                   exactly=False)
        self.quotes_frame = self.quotes_frame.reset_index()

        if self.timeit:
            self.do_time()

    def quotes_percent(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Evaluating Quotes Percents')

        self.quotes_frame = self.quotes_frame.set_index(keys=['ticker', 'time'])
        self.quotes_frame = self.quotes_frame.sort_index(ascending=True)
        self.quotes_frame = percent_it(frame=self.quotes_frame, horizon=1)
        self.quotes_frame = self.quotes_frame.reset_index()

        if self.timeit:
            self.do_time()

    def time_features(self, the_data):

        if self.add_time_features:
            from busy_exchange.utils import BusyDayExchange, BusyTimeExchange

            """
            the_data['time'] = the_data['time'].dt.tz_convert('EST')

            the_data['is_holi'] = the_data['time'].apply(func=BusyDayExchange.is_holi).astype(dtype=float)
            the_data['is_full'] = the_data['time'].apply(func=BusyDayExchange.is_full).astype(dtype=float)
            the_data['is_cut'] = the_data['time'].apply(func=BusyDayExchange.is_cut).astype(dtype=float)

            the_data['to_holi'] = the_data['time'].apply(func=BusyDayExchange.to_holiday, args=(True,))
            the_data['to_full'] = the_data['time'].apply(func=BusyDayExchange.to_fullday, args=(True,))
            the_data['to_cut'] = the_data['time'].apply(func=BusyDayExchange.to_cutday, args=(True,))
            the_data['af_holi'] = the_data['time'].apply(func=BusyDayExchange.to_holiday, args=(False,))
            the_data['af_full'] = the_data['time'].apply(func=BusyDayExchange.to_fullday, args=(False,))
            the_data['af_cut'] = the_data['time'].apply(func=BusyDayExchange.to_cutday, args=(False,))
            """
            the_data['mday'] = the_data['time'].dt.day
            the_data['wday'] = the_data['time'].dt.dayofweek
            the_data['hour'] = the_data['time'].dt.hour
            the_data['minute'] = the_data['time'].dt.minute

            # the_data['to_open'] = the_data['time'].apply(func=BusyTimeExchange.to_open)
            # the_data['to_close'] = the_data['time'].apply(func=BusyTimeExchange.to_close)

        return the_data

    async def read(self):

        if self.verbose:
            print('Reading')

        self.establish_connection()
        self.prepare_news_titles_frame()
        await self.prepare_quotes()
        self.quotes_fill()
        self.quotes_lag()
        # self.quotes_percent()

        if self.base_option == 'for_merge':

            self.quotes_frame.to_sql(name=self.quotes_alias, con=self.connection,
                                                  if_exists='replace',
                                                  index=False)

            query = """
            CREATE TEMPORARY TABLE {0} AS
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
                    public.{1} AS QD
                    ON NF."time" = QD."time") AS RS
                 WHERE 37 = 37
             ;
             """.format(self.result_alias, self.quotes_alias)

            self.connection.execute(query)

            if self.export_chunk is None:

                reader_query = """
                SELECT *
                FROM {0}
                ;
                """.format(self.result_alias)
                the_data = pandas.read_sql(sql=reader_query, con=self.connection)

                the_data = self.time_features(the_data)

                if self.nlp_treator is not None and self.nlp_ductor == 'post':
                    old_name = 'title'
                    new_name = 'Text'
                    the_data['title'] = the_data['title'].fillna('NoData')
                    print('HUGO BOSS: to memory')
                    the_data = the_data.rename(columns={old_name: new_name})
                    the_data = self.nlp_treator(the_data,
                                                self.nlp_treator_signature, self.nlp_treator_config)

                    the_data = the_data.rename(columns={new_name: old_name})

                    return the_data

            else:

                size_query = """ 
                SELECT COUNT(*)
                FROM {0}
                ;
                """.format(self.result_alias)
                final_table_d0_size = pandas.read_sql(sql=size_query, con=self.connection).values[0, 0]
                n_chunks = (final_table_d0_size // self.export_chunk) + 1
                chunks = [(j * self.export_chunk, (j + 1) * self.export_chunk - 1) for j in range(n_chunks)]
                chunks[-1] = (chunks[-1][0], final_table_d0_size)

                if self.verbose:
                    print("Final table's D0:\t {0}\nChunks:\n{1}".format(final_table_d0_size, chunks))

                iteration_columns_query = """
                ALTER TABLE {0}
                ADD COLUMN chunker SERIAL; 
                """.format(self.result_alias)

                self.connection.execute(iteration_columns_query)

                if os.path.exists(self.where_to_save):
                    os.remove(self.where_to_save)

                for j in range(n_chunks):

                    reader_query = """
                    SELECT *
                    FROM {0}
                    WHERE chunker >= {1} and chunker <= {2}
                    ;
                    """.format(self.result_alias, chunks[j][0], chunks[j][1])

                    data_chunk = pandas.read_sql(sql=reader_query, con=self.connection)

                    data_chunk = self.time_features(data_chunk)

                    if self.nlp_treator is not None and self.nlp_ductor == 'post':
                        old_name = 'title'
                        new_name = 'Text'
                        data_chunk['title'] = data_chunk['title'].fillna('NoData')
                        print('HUGO BOSS: to disk')
                        data_chunk = data_chunk.rename(columns={old_name: new_name})
                        data_chunk = self.nlp_treator(data_chunk,
                                                    self.nlp_treator_signature, self.nlp_treator_config)

                        data_chunk = data_chunk.rename(columns={new_name: old_name})

                    data_chunk.columns = [x.replace('_PCT1', '') for x in data_chunk.columns.values]
                    data_chunk = data_chunk.dropna().sort_values(by=['title', 'lag'])

                    if j == 0:
                        data_chunk.to_csv(self.where_to_save, sep=';', index=False, mode='a', header=True)
                    else:
                        data_chunk.to_csv(self.where_to_save, sep=';', index=False, mode='a', header=False)

        else:

            the_data = self.quotes_frame.merge(right=self.news_titles_frame, left_on='time', right_on='time')

            the_data = self.time_features(the_data)

            if self.nlp_treator is not None and self.nlp_ductor == 'post':
                old_name = 'title'
                new_name = 'Text'
                the_data['title'] = the_data['title'].fillna('NoData')
                print('HUGO BOSS: in memory')
                the_data = the_data.rename(columns={old_name: new_name})
                the_data = self.nlp_treator(the_data,
                                            self.nlp_treator_signature, self.nlp_treator_config)

                the_data = the_data.rename(columns={new_name: old_name})

            return the_data


class KernelLoader:

    def __init__(self, api_key, target_quotes, news_horizon, effect_horizon, db_config,
                 window_function, window_function_kwargs,
                 reload_quotes=False,
                 news_titles_source=None, verbose=False, timeit=False, base_option='for_merge', add_time_features=False,
                 nlp_treator=None, nlp_treator_signature=None, nlp_treator_config=None, nlp_ductor='post',
                 export_chunk=100_000):

        self.window_function = window_function
        self.window_function_kwargs = window_function_kwargs

        self.verbose = verbose
        self.timeit = timeit
        self.run_time = None
        self.base_option = base_option
        self.add_time_features = add_time_features

        self.export_chunk = export_chunk
        self.where_to_save = './result.csv'

        self.nlp_treator = nlp_treator
        self.nlp_treator_signature = nlp_treator_signature
        self.nlp_treator_config = nlp_treator_config
        self.nlp_ductor = nlp_ductor

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

        self.news_titles_alias = 'news_titles'
        self.quotes_alias = 'quotes'
        self.result_alias = 'result_table'

    def fix_time(self):

        self.run_time = time.time()

    def do_time(self):

        self.run_time = time.time() - self.run_time
        print(self.run_time)

    def establish_connection(self):

        if self.timeit:
            self.fix_time()

        if self.base_option == 'for_merge':

            if self.verbose:
                print('Establishing Connection')

            with open(self.db_config) as f:
                db_config = json.load(f)

            user, password, host, port, dbname = db_config['user'], db_config['password'], db_config['host'], db_config[
                'port'], db_config['dbname']

            connection_string = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(user, password, host, port, dbname)
            engine = sqlalchemy.create_engine(connection_string)

            self.connection = engine.connect()

            if self.timeit:
                self.do_time()

        else:

            if self.verbose:
                print('Skipped Connection')

    def prepare_news_titles_frame(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Preparing News Titles Frame')

        if self.news_titles_source is None:
            raise Exception("You should specify news titles source")

        if self.news_titles_source is not None:

            self.news_titles_frame = pandas.read_excel(self.news_titles_source)

            self.news_titles_frame['time'] = pandas.to_datetime(self.news_titles_frame['time'])

            def fix_tz(x):
                return x.tz_localize(tz='UTC')

            self.news_titles_frame['time'] = self.news_titles_frame['time'].apply(func=fix_tz)

            def fixit(x):
                return x.ceil(freq='T')

            self.news_titles_frame['time'] = self.news_titles_frame['time'].apply(func=fixit)

            if self.nlp_treator is not None:  # and self.nlp_ductor == 'pre':
                old_name = 'title'
                new_name = 'Text'
                self.news_titles_frame = self.news_titles_frame.rename(columns={old_name: new_name})
                self.news_titles_frame = self.nlp_treator(self.news_titles_frame,
                                                          self.nlp_treator_signature, self.nlp_treator_config)

                self.news_titles_frame = self.news_titles_frame.rename(columns={new_name: old_name})

            # self.news_titles_frame = self.news_titles_frame[['id', 'time', 'title']]
            self.news_titles_frame = self.news_titles_frame.drop(columns=['source', 'category'])

            """
            lag_markers = list(
                itertools.product(self.news_titles_frame['id'].values,
                                  numpy.array(numpy.arange(self.news_horizon - 1)) + 1))
            lag_markers = pandas.DataFrame(data=lag_markers, columns=['id', 'lag'])
            self.news_titles_frame = self.news_titles_frame.merge(right=lag_markers, left_on=['id'],
                                                                  right_on=['id'])
            """

            def minute_offset(x):
                return pandas.DateOffset(minutes=x)

            self.news_titles_frame['time'] = pandas.to_datetime(self.news_titles_frame['time'])
            """
            self.news_titles_frame['news_time'] = self.news_titles_frame['time'].copy()
            self.news_titles_frame['time'] = self.news_titles_frame['lag'].apply(func=minute_offset)
            self.news_titles_frame['time'] = self.news_titles_frame['news_time'] + self.news_titles_frame['time']

            if self.base_option == 'for_merge':
                self.news_titles_frame.to_sql(name=self.news_titles_alias, con=self.connection,
                                              if_exists='replace',
                                              index=False)
            """
        if self.timeit:
            self.do_time()

    def get_dates(self):

        if self.verbose:
            print('Getting Dates')

        if self.base_option == 'without':

            beginning_date, ending_date = self.news_titles_frame['time'].min() - pandas.DateOffset(
                minutes=self.effect_horizon), self.news_titles_frame['time'].max()

        else:

            beginning_date_query = """
            SELECT (MIN(time) - ({1} * INTERVAL '1 minute')) AS mn
            FROM {0}
            """.format(self.news_titles_alias, self.effect_horizon)

            ending_date_query = """
            SELECT MAX(time) as mx
            FROM {0}
            """.format(self.news_titles_alias)

            beginning_date = pandas.read_sql(sql=beginning_date_query, con=self.connection).values[0, 0]
            ending_date = pandas.read_sql(sql=ending_date_query, con=self.connection).values[0, 0]

        return beginning_date, ending_date

    async def call_quotes(self):

        beginning_date, ending_date = self.get_dates()

        self.quotes_frame = await call_them_all(tickers=self.target_quotes, start_date=beginning_date,
                                                end_date=ending_date, token=self.api_key)

    async def prepare_quotes(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Preparing Quotes')

        if self.reload_quotes:

            await self.call_quotes()

        if self.timeit:
            self.do_time()

    def quotes_fill(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Filling Quotes')

        beginning_date, ending_date = self.get_dates()

        self.quotes_frame = self.quotes_frame.set_index(keys=['ticker', 'time'])
        self.quotes_frame = self.quotes_frame.sort_index(ascending=True)
        self.quotes_frame = fill_it(frame=self.quotes_frame, freq='T', zero_index_name='ticker',
                                    first_index_name='time')
        self.quotes_frame = self.quotes_frame.reset_index()

        if self.timeit:
            self.do_time()

    def quotes_lag(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Lagging Quotes')

        self.quotes_frame = self.quotes_frame.set_index(keys=['ticker', 'time'])
        self.quotes_frame = self.quotes_frame.sort_index(ascending=True)
        self.quotes_frame = lag_it(frame=self.quotes_frame, n_lags=self.effect_horizon, suffix='_LAG',
                                   exactly=False)
        self.quotes_frame = self.quotes_frame.reset_index()

        if self.timeit:
            self.do_time()

    def quotes_percent(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Evaluating Quotes Percents')

        self.quotes_frame = self.quotes_frame.set_index(keys=['ticker', 'time'])
        self.quotes_frame = self.quotes_frame.sort_index(ascending=True)
        self.quotes_frame = percent_it(frame=self.quotes_frame, horizon=1)
        self.quotes_frame = self.quotes_frame.reset_index()

        if self.timeit:
            self.do_time()

    def time_features(self, the_data):

        if self.add_time_features:
            from busy_exchange.utils import BusyDayExchange, BusyTimeExchange

            """
            the_data['time'] = the_data['time'].dt.tz_convert('EST')

            the_data['is_holi'] = the_data['time'].apply(func=BusyDayExchange.is_holi).astype(dtype=float)
            the_data['is_full'] = the_data['time'].apply(func=BusyDayExchange.is_full).astype(dtype=float)
            the_data['is_cut'] = the_data['time'].apply(func=BusyDayExchange.is_cut).astype(dtype=float)

            the_data['to_holi'] = the_data['time'].apply(func=BusyDayExchange.to_holiday, args=(True,))
            the_data['to_full'] = the_data['time'].apply(func=BusyDayExchange.to_fullday, args=(True,))
            the_data['to_cut'] = the_data['time'].apply(func=BusyDayExchange.to_cutday, args=(True,))
            the_data['af_holi'] = the_data['time'].apply(func=BusyDayExchange.to_holiday, args=(False,))
            the_data['af_full'] = the_data['time'].apply(func=BusyDayExchange.to_fullday, args=(False,))
            the_data['af_cut'] = the_data['time'].apply(func=BusyDayExchange.to_cutday, args=(False,))
            """
            the_data['mday'] = the_data['time'].dt.day
            the_data['wday'] = the_data['time'].dt.dayofweek
            the_data['hour'] = the_data['time'].dt.hour
            the_data['minute'] = the_data['time'].dt.minute

            # the_data['to_open'] = the_data['time'].apply(func=BusyTimeExchange.to_open)
            # the_data['to_close'] = the_data['time'].apply(func=BusyTimeExchange.to_close)

        return the_data

    async def read(self):

        if self.verbose:
            print('Reading')

        self.establish_connection()
        self.prepare_news_titles_frame()
        await self.prepare_quotes()
        self.quotes_fill()

        vol_cols = [x for x in self.quotes_frame.columns.values if 'volume' in x]
        self.quotes_frame[vol_cols] = self.quotes_frame[vol_cols].fillna(value=0)
        self.quotes_frame = self.quotes_frame.fillna(method='pad')

        self.quotes_lag()
        # self.quotes_percent()

        self.news_titles_frame = self.news_titles_frame.drop(columns=['id', 'title'])
        self.news_titles_frame = self.news_titles_frame.groupby(by='time').mean()

        news_titles_columns = [x for x in self.news_titles_frame.columns.values if x != 'time']
        result = self.quotes_frame.merge(right=self.news_titles_frame, how='left', left_on='time', right_index=True) # right_on='time')

        use_cols = [x for x in result.columns.values if 'USE' in x]
        result[use_cols] = result[use_cols].fillna(value=0)

        # here go these windows
        if self.window_function == 'ewm':
            result[news_titles_columns] = result[news_titles_columns].ewm(**self.window_function_kwargs).mean()
        elif self.window_function == 'rolling':
            result[news_titles_columns] = result[news_titles_columns].rolling(**self.window_function_kwargs).mean()
        else:
            raise Exception("Misunderstood windows function")

        return result
