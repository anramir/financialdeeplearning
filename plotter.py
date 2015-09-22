__author__ = 'Andres'
from financialdata import StockData

name='acs'
data = StockData()
data.readData('stockprices/'+name+'.txt')
data.normalizeData()
data.delayInputs()
data.plotData()