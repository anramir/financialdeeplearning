__author__ = 'Andres'

import datetime
from financialdata import StockSample

start=datetime.date(2010,05,01)
end=datetime.date(2015,05,01)
collapse='daily'

stocks = []

santander=StockSample('santander',collapse,start,end)

stocks.append(santander)

for i in range(len(stocks)):
    stocks[i].normalizeData()
    stocks[i].calculateReturns()
    stocks[i].forecastReturn()
    stocks[i].plotPredictions()

