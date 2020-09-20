#
import asyncio
from data.flat import load

k = 'C:/Users/MainUser/Desktop/OpenAPI_sandbox.txt'
crs = open(k, "r")
for columns in ( raw.strip().split() for raw in crs ):
    api_key = columns[0]

target_quotes = ['MSFT']
news_horizon = 10
effect_horizon = 1
max_quotes_lag = 10

data = asyncio.run(load(api_key, target_quotes, news_horizon, effect_horizon, max_quotes_lag, show_shapes=True))
print(data)
