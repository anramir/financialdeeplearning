__author__ = 'Andres'
import Quandl
import numpy as np
from pybrain.datasets.sequential import SequentialDataSet
from financialmodel import FinancialTrainedNetwork
import matplotlib.pyplot as plt
from pybrain.tools.customxml.networkreader import NetworkReader
from numpy import zeros, array, append
import pylab

ibexStocks={'abertis':'YAHOO/MC_ABE',
            'acciona':'YAHOO/MC_ANA',
            'acerinox':'YAHOO/MC_ACX',
            'acs':'YAHOO/MC_ACS',
            'bankinter':'YAHOO/MC_BKT',
            'bbva':'YAHOO/MC_BBVA',
            'caixabank':'YAHOO/MC_CABK',
            'dia':'YAHOO/MC_DIA',
            'enagas':'YAHOO/MC_ENG',
            'endesa':'YAHOO/MC_ELE',
            'ferrovial':'YAHOO/MC_FER',
            'grifolsa':'YAHOO/MC_GRF',
            'inditex':'YAHOO/MC_ITX',
            'jazztel':'YAHOO/MC_JAZ',
            'mapfre':'YAHOO/MC_MAP',
            'ohl':'YAHOO/MC_OHL',
            'repsol':'YAHOO/MC_REP',
            'santander':'YAHOO/MC_SAN',
            'telefonica':'YAHOO/MC_TEF',
            'viscofan':'YAHOO/MC_VIS'}


class StockData:
    def __init__(self):
        self.data=[]
        self.trainData=[]
        self.testData=[]

    def downloadData(self,stock,collapse,start="2012-12-01",end="2013-01-01"):
        self.stock=stock
        self.start=start
        self.end=end
        self.data = Quandl.get(ibexStocks[self.stock], authtoken="4bosWLqsiGqMtuuuYAcq", collapse=collapse, trim_start=self.start, trim_end=self.end, returns='numpy')


    def saveData(self,name):
        with open (name,'w') as f:
            for i in range(len(self.data)):
                if self.data[i][5]:
                    f.write("%.3f\t%.3f\t%.3f\t%.3f\t%d\t%.3f\n" % (self.data[i][1],self.data[i][2],self.data[i][3],self.data[i][4],self.data[i][5], self.data[i][4]))
                else:
                    pass


    def readData(self,name,delimiter='\t'):
        with open(name) as f:
            for line in f:
                self.data.append(line.strip().split(delimiter))

        for item in self.data:
            for i in range(len(item)):
                item[i]=float(item[i])

        self.data=np.array(self.data)



    def normalizeData(self):
        def normalize(vector):
            maximo=max(vector)
            for i in range(len(vector)):
                vector[i]=vector[i]/maximo

            return vector
        for i in range(self.data.shape[1]):
            self.data[:,i]=normalize(self.data[:,i])

    def delayInputs(self):
        m=len(self.data)
        for i in range(1,m):
            self.data[i-1,-1]=self.data[i,-1]
        self.data=np.delete(self.data,m-1,axis=0)


    def createSequentialDataSets(self,testRatio=0.7):
        ixSeparator=int(self.data.shape[0]*0.7)
        trainData=self.data[0:ixSeparator]
        testData=self.data[ixSeparator:]

        self.trainData = SequentialDataSet(5,1)
        self.testData= SequentialDataSet(5,1)

        for i in range(len(trainData)):
            self.trainData.addSample(trainData[i,0:5],trainData[i,5])

        for i in range(len(testData)):
            self.testData.addSample(testData[i,0:5],testData[i,5])

    def plotData(self):
        plt.plot(self.data[:,5],'b')
        pylab.show()


class StockSample:
    def __init__(self,stock,collapse,start,end):
        self.stock=stock
        data = Quandl.get(ibexStocks[stock], authtoken="4bosWLqsiGqMtuuuYAcq", collapse=collapse, trim_start=start, trim_end=end, returns='numpy')
        self.data=[]
        for i in range(len(data)):
            row=[]
            for j in range(1,6):
                row.append(data[i][j])
            self.data.append(row)
        self.data=np.asarray(self.data)


    def normalizeData(self):
        def normalize(vector):
            maximo=max(vector)
            for i in range(len(vector)):
                vector[i]=vector[i]/maximo

            return vector
        for i in range(self.data.shape[1]):
            self.data[:,i]=normalize(self.data[:,i])

    def calculateReturns(self):
        self.returns=[]
        for i in range(1,len(self.data)):
            self.returns.append((self.data[i,3]-self.data[i-1,3])/self.data[i-1,3])

    def forecastReturn(self):
        self.expectedPrices=[]
        model=FinancialTrainedNetwork(self.stock)
        wtratio=1./3.
        wtseparator=int(wtratio*self.data.shape[0])
        washout_inputs=self.data[:wtseparator]

        self.activation_inputs=self.data[wtseparator:]

        model.activate(washout_inputs)
        self.expectedPrices=model.activate(self.activation_inputs)
        self.expectedReturn=(self.expectedPrices[-1]-self.activation_inputs[-1,3])/self.activation_inputs[-1,3]

    def plotPredictions(self):
        data=self.expectedPrices
        data.insert(1,0)
        plt.plot(data[1:],'b',self.activation_inputs[1:,3],'r')
        plt.show()

    def _getLastOutput(self):
        if self.model.offset == 0:
            return zeros(self.model.outdim)
        else:
            return self.model._out_layer.outputbuffer[self.network.offset - 1]