from financialdata import StockData
from financialmodel import FinancialNetwork
from evolinotrainer import EvolinoTrainer, EvolinoEvaluation
from pylab import plot, show, ion, cla, subplot, title, figlegend, draw
from pybrain.tools.customxml.networkwriter import NetworkWriter
import numpy as np
import matplotlib.pyplot as plt
import pylab
name='acs'

data=StockData()
data.readData('stockprices/'+name+'.txt')
data.normalizeData()
data.delayInputs()
data.createSequentialDataSets()

model = FinancialNetwork(data.trainData.indim,data.trainData.outdim,hiddim=10)

trainer=EvolinoTrainer(model,data.trainData,
    subPopulationSize = 20,
    nParents = 8,
    nCombinations = 2,
    initialWeightRange = ( -0.1 , 0.1 ),
    mutationAlpha = 0.001,
    nBurstMutationEpochs = np.Infinity,
    verbosity = 2)

trainInput = data.trainData.getField('input')
testInput = data.testData.getField('input')

trainTarget = data.trainData.getField('target')
testTarget = data.testData.getField('target')

wtRatio=1./3.

training_washout_separator=int(wtRatio*len(trainInput))
test_washout_separator=int(wtRatio*len(testInput))

washout_trainInput=trainInput[:training_washout_separator]
washout_testInput=testInput[:test_washout_separator]
forecast_trainInput=trainInput[training_washout_separator:]
forecast_testInput=testInput[test_washout_separator:]

target_trainInput=trainTarget[training_washout_separator:]
target_testInput=testTarget[test_washout_separator:]



ion() # switch matplotlib to interactive mode
for i in range(100):
    print("======================")
    print("====== NEXT RUN ======")
    print("======================")

    print("=== TRAINING")
    # train the network for 1 epoch
    trainer.trainEpochs( 1 )


    print("=== PLOTTING\n")
    # calculate the nets output for train and the test data

    model.activate(washout_trainInput)
    trnSequenceOutput = model.activate(forecast_trainInput)

    model.activate(washout_testInput)
    tstSequenceOutput = model.activate(forecast_testInput)

    NetworkWriter.writeToFile(model.network,'trainednetworks/'+name+'.xml')
    with open ('trainednetworks/trainresults/%s_train_results.txt'% (name),'w') as f:
        for i in range(len(trnSequenceOutput)):
            f.write('%.7f\t%.7f\n' % (target_trainInput[i],trnSequenceOutput[i]))
        f.close()

    with open ('trainednetworks/trainresults/%s_test_results.txt'% (name),'w') as f:
        for i in range(len(tstSequenceOutput)):
            f.write('%.7f\t%.7f\n' % (target_testInput[i],tstSequenceOutput[i]))
        f.close()

    # plot training data
    sp = subplot(211) # switch to the first subplot
    cla() # clear the subplot
    targetline = plot(target_trainInput,"r-") # plot the targets
    outputline = plot(trnSequenceOutput,"b-") # plot the actual output


    # plot test data
    sp = subplot(212)
    cla()
    plot(target_testInput,"r-")
    plot(tstSequenceOutput,"b-")

    # draw everything
    draw()
show()


