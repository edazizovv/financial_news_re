#
import json
import pandas
import sqlalchemy


#


#
def get_connection():

    with open('C:/Users/MainUser/Desktop/users.json') as f:
        users = json.load(f)

    user, password = users['researcher']['user'], users['researcher']['password']

    with open('C:/Users/MainUser/Desktop/servers.json') as f:
        servers = json.load(f)

    host, port = servers['GOLA']['host'], servers['GOLA']['port']

    dbname = 'experiments'

    connection_string = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(user, password, host, port, dbname)
    engine = sqlalchemy.create_engine(connection_string)
    connection = engine.connect()

    return connection


def reported_n_muted(report_measures, report_values):

    connection = get_connection()

    report_measures.to_sql(name='report_measures', con=connection, if_exists='append', index=False)
    report_values.to_sql(name='report_values', con=connection, if_exists='append', index=False)


def get_all_stuff():

    connection = get_connection()

    report_measures = pandas.read_sql(sql="SELECT * FROM report_measures;", con=connection)
    report_values = pandas.read_sql(sql="SELECT * FROM report_values;", con=connection)

    return report_measures, report_values
