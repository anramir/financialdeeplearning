__author__ = 'Andres'
from financialdata import StockData

start = "2010-01-02"
end = "2015-01-01"
name='acs'
collapse='daily'


data = StockData()
data.downloadData(name,collapse,start=start,end=end)
data.saveData('stockprices/'+name+'.txt')