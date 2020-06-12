#
import time
import pandas
import pandas_datareader
import datetime
import itertools
import numpy
import json
import sqlalchemy

from tinkoff_api.quotes_loader import call_them_all
from m_utils.transform import lag_it, percent_it, fill_it


def sql_formatting(x):
    return str(x).replace('[', '').replace(']', '').replace("'", '')


class Loader:

    def __init__(self, api_key, target_quotes, news_horizon, effect_horizon, db_config, reload_quotes=False,
                 news_titles_source=None, verbose=False, timeit=False, base_option='full'):

        self.verbose = verbose
        self.timeit = timeit
        self.run_time = None
        self.base_option = base_option

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

    def fix_time(self):

        self.run_time = time.time()

    def do_time(self):

        self.run_time = time.time() - self.run_time
        print(self.run_time)

    def establish_connection(self):

        if self.timeit:
            self.fix_time()

        if self.base_option in ['full', 'for_merge']:

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

        if self.news_titles_source is None and self.base_option != 'full':

            raise Exception("Sorry, I do not know how are you going to use stored news titles without a database")

        if self.news_titles_source is not None:

            self.news_titles_frame = pandas.read_excel(self.news_titles_source)

            self.news_titles_frame['time'] = pandas.to_datetime(self.news_titles_frame['time'])

            def fix_tz(x):
                return x.tz_localize(tz='UTC')

            self.news_titles_frame['time'] = self.news_titles_frame['time'].apply(func=fix_tz)

            def fixit(x):
                return x.ceil(freq='T')

            self.news_titles_frame['time'] = self.news_titles_frame['time'].apply(func=fixit)

            if self.base_option == 'full':

                self.news_titles_frame.to_sql(name=self.news_titles_raw_alias, con=self.connection, if_exists='replace',
                                              index=False)

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

            else:

                self.news_titles_frame = self.news_titles_frame[['id', 'time', 'title']]
                lag_markers = list(
                    itertools.product(self.news_titles_frame['id'].values, numpy.array(numpy.arange(self.news_horizon - 1)) + 1))
                lag_markers = pandas.DataFrame(data=lag_markers, columns=['id', 'lag'])
                self.news_titles_frame = self.news_titles_frame.merge(right=lag_markers, left_on=['id'], right_on=['id'])

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

            if self.base_option == 'full':
                table = self.news_titles_raw_alias
            if self.base_option == 'for_merge':
                table = self.news_titles_alias

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

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Preparing Quotes')

        if self.reload_quotes:

            await self.call_quotes()

            if self.base_option == 'full':

                self.quotes_frame.to_sql(name=self.quotes_alias, con=self.connection, if_exists='replace', index=False)

        if self.timeit:
            self.do_time()

    def quotes_fill(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Filling Quotes')

        beginning_date, ending_date = self.get_dates()

        if self.base_option == 'full':

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
            """.format(self.quotes_alias, sql_formatting(cc), self.mid_name)
            self.connection.execute(mid_query)

        else:

            self.quotes_frame = self.quotes_frame.set_index(keys=['ticker', 'time'])
            self.quotes_frame = self.quotes_frame.sort_index(ascending=True)
            self.quotes_frame = fill_it(frame=self.quotes_frame, freq='T', zero_index_name='ticker', first_index_name='time')
            self.quotes_frame = self.quotes_frame.reset_index()

        if self.timeit:
            self.do_time()

    def quotes_lag(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Lagging Quotes')

        if self.base_option == 'full':

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
                """.format(self.mid_name, column, lag_alias, n_lags, sql_formatting(identifiers), sql_formatting(identifiers_and_time))
                self.connection.execute(query_execute)

        else:

            self.quotes_frame = self.quotes_frame.set_index(keys=['ticker', 'time'])
            self.quotes_frame = self.quotes_frame.sort_index(ascending=True)
            self.quotes_frame = lag_it(frame=self.quotes_frame, n_lags=self.effect_horizon, suffix='_LAG', exactly=False)
            self.quotes_frame = self.quotes_frame.reset_index()

        if self.timeit:
            self.do_time()

    def quotes_percent(self):

        if self.timeit:
            self.fix_time()

        if self.verbose:
            print('Evaluating Quotes Percents')

        if self.base_option == 'full':

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
                """.format(self.mid_name, column, sql_formatting(identifiers), sql_formatting(identifiers_and_time))

                self.connection.execute(query_execute)

        else:

            self.quotes_frame = self.quotes_frame.set_index(keys=['ticker', 'time'])
            self.quotes_frame = self.quotes_frame.sort_index(ascending=True)
            self.quotes_frame = percent_it(frame=self.quotes_frame, horizon=1)
            self.quotes_frame = self.quotes_frame.reset_index()

        if self.timeit:
            self.do_time()

    async def read(self):

        if self.verbose:
            print('Reading')

        self.establish_connection()
        self.prepare_news_titles_frame()
        await self.prepare_quotes()
        self.quotes_fill()
        self.quotes_lag()
        self.quotes_percent()

        if self.base_option in ['full', 'for_merge']:

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

        else:

            the_data = self.quotes_frame.merge(right=self.news_titles_frame, left_on='time', right_on='time')

        return the_data
