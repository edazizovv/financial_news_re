#
import json
import time
import numpy
import pandas
import asyncio
from data.flat import KernelLoader


#
from news_embedder.overhelm import embedding_pool
from news_embedder.configuration import Config


#
k = 'C:/Users/MainUser/Desktop/OpenAPI_sandbox.txt'
crs = open(k, "r")
api_key = 'alive_outside'
for columns in (raw.strip().split() for raw in crs):
    api_key = columns[0]

target_quotes = ['MSFT']
news_horizon = 100
effect_horizon = 100

db_config = 'C:/Users/MainUser/Desktop/config.json'
news_titles_source = './data/data/rex.xlsx'

config = Config()
config.model = {'agg': 'mean'}

loader = KernelLoader(api_key, target_quotes, news_horizon, effect_horizon, db_config,
                      window_function='ewm', window_function_kwargs={'alpha': 0.1, 'adjust': False},
                      reload_quotes=True,
                      news_titles_source=news_titles_source, verbose=True, base_option='for_merge', add_time_features=True,
                      nlp_treator=embedding_pool, nlp_treator_signature=['sister'], nlp_treator_config=config,
                      nlp_ductor='post', export_chunk=100_000)
# base_option='without'

# data = await loader.read()
data = asyncio.run(loader.read())
"""
data = data.dropna()

d = './result.csv'
data.to_csv(d, sep=';', index=False)
"""
"""
# data.columns = [x.replace('_PCT1', '') for x in data.columns.values]
data.dropna().sort_values(by=['title', 'lag'])

d = './dataset_use_timed.csv'
data.dropna().sort_values(by=['title', 'lag']).to_csv(d, index=False)
"""
"""
g = 'E:/dataset_use_timed.csv'
data = pandas.read_csv(g, sep=';')
"""
