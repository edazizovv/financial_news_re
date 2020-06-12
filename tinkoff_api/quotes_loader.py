#
import pandas
import datetime
from tinkoff.investments.client.exceptions import TinkoffInvestmentsError
from tinkoff.investments import TinkoffInvestmentsRESTClient, Environment, CandleResolution


# https://github.com/Fatal1ty/tinkoff-api
async def show_my_time_candles(ticker, token, start_date, end_date, interval=CandleResolution.MIN_1):
    try:
        async with TinkoffInvestmentsRESTClient(
                token=token,
                environment=Environment.SANDBOX) as client:

            instruments = await client.market.instruments.search(ticker)
            instrument = instruments[0]
            figi = instrument.figi

            candles = await client.market.candles.get(
                figi=figi,
                dt_from=start_date,
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
            start_part = start_date + datetime.timedelta(days=dd)
            end_part = start_date + datetime.timedelta(days=(dd + 1))
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

