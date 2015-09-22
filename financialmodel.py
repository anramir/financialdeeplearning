__author__ = 'Andres'

from pybrain.structure.modules.lstm         import LSTMLayer
from pybrain.structure.modules.linearlayer  import LinearLayer
from pybrain.structure.connections.full     import FullConnection
from pybrain.structure.modules.biasunit     import BiasUnit
from pybrain.structure.networks.recurrent import RecurrentNetwork
from numpy import zeros, array, append
from numpy import reshape
from copy  import copy, deepcopy
from pybrain.tools.customxml.networkreader import NetworkReader

class FinancialNetwork(object):
    def __init__(self,indim,outdim,hiddim=6):
        self.network=RecurrentNetwork()

        ##CREATE MODULES
        self._in_layer = LinearLayer(indim+outdim)
        self._hid_layer = LSTMLayer(hiddim,peepholes=False)
        self._out_layer = LinearLayer(outdim)
        self._bias = BiasUnit()

        ##ADD MODULES
        self.network.addInputModule(self._in_layer)
        self.network.addModule(self._hid_layer)
        self.network.addModule(self._bias)
        self.network.addOutputModule(self._out_layer)
        self._last_hidden_layer = None
        self._first_hidden_layer = None

        ###CREATE CONNECTIONS
        self._hid_to_out_connection = FullConnection(self._hid_layer , self._out_layer)
        self._in_to_hid_connection = FullConnection(self._in_layer  , self._hid_layer)
        self._out_to_hid_connection=FullConnection(self._out_layer,self._hid_layer)

        ##ADD CONNECTIONS
        self.network.addConnection(self._hid_to_out_connection)
        self.network.addConnection(self._in_to_hid_connection)
        self.network.addConnection(FullConnection(self._bias, self._hid_layer))
        self.network.addRecurrentConnection(self._out_to_hid_connection)
        self.network.sortModules()

        self.backprojectionFactor = 1



    def getGenome(self):
        weights = []
        for layer in self.getHiddenLayers():
            if isinstance(layer, LSTMLayer):
#                 if layer is not self._recurrence_layer:
                weights += self._getGenomeOfLayer(layer)
        return weights

    def setGenome(self, weights):
        """ Sets the Genome of the network.
            See class description for more details.
        """
        weights = deepcopy(weights)
        for layer in self.getHiddenLayers():
            if isinstance(layer, LSTMLayer):
#               if layer is not self._recurrence_layer:
                self._setGenomeOfLayer(layer, weights)

    def _setGenomeOfLayer(self, layer, weights):

        dim = layer.outdim

        connections = self._getInputConnectionsOfLayer(layer)

        for cell_idx in range(dim):
            cell_weights = weights.pop(0)
            for c in connections:
                params = c.params
                params[cell_idx + 0 * dim] = cell_weights.pop(0)
                params[cell_idx + 1 * dim] = cell_weights.pop(0)
                params[cell_idx + 2 * dim] = cell_weights.pop(0)
                params[cell_idx + 3 * dim] = cell_weights.pop(0)
            assert not len(cell_weights)

    def _getGenomeOfLayer(self, layer):

        dim = layer.outdim
        layer_weights = []

        connections = self._getInputConnectionsOfLayer(layer)

        for cell_idx in range(dim):
            # todo: the evolino paper uses a different order of weights for the genotype of a lstm cell
            cell_weights = []
            for c in connections:
                cell_weights += [
                    c.params[ cell_idx + 0 * dim ],
                    c.params[ cell_idx + 1 * dim ],
                    c.params[ cell_idx + 2 * dim ],
                    c.params[ cell_idx + 3 * dim ] ]

            layer_weights.append(cell_weights)
        return layer_weights

    def _getInputConnectionsOfLayer(self, layer):
        """ Returns a list of all input connections for the layer. """
        connections = []
        for c in sum(list(self.network.connections.values()), []):
            if c.outmod is layer:
                if not isinstance(c, FullConnection):
                    raise NotImplementedError("At the time there is only support for FullConnection")
                connections.append(c)
        return connections


    def getHiddenLayers(self):
        """ Returns a list of all hidden layers. """
        layers = []
        network = self.network
        for m in network.modules:
            if m not in network.inmodules and m not in network.outmodules:
                layers.append(m)
        return layers


    def _getLastOutput(self):
        if self.network.offset == 0:
            return zeros(self.network.outdim)
        else:
            return self._out_layer.outputbuffer[self.network.offset - 1]

    def _setLastOutput(self, output):
        self._out_layer.outputbuffer[self.network.offset - 1][:] = output


    def washout(self,input):
        self.network.offset=0
        lstmvalues=[]
        for val in input:
            backprojection=self._getLastOutput()
            backprojection*=self.backprojectionFactor
            input=append(val,backprojection)
            output=self.network.activate(input)
            self._setLastOutput(output)
            lstmvalues.append(self._hid_layer.outputbuffer[self.network.offset - 1])

        return lstmvalues

    def reset(self):
        self.network.reset()


    def getOutputWeightMatrix(self):
        c=self._hid_to_out_connection
        W=c.params
        return reshape(W, (c.outdim, c.indim))

    def setOutputWeightMatrix(self,W):
        c=self._hid_to_out_connection
        p=c.params
        p[:]=W.flatten()

    def activate(self,input):
        outputs=[]
        for val in input:
            backprojection=self._getLastOutput()
            backprojection*=self.backprojectionFactor
            inputtrain=append(val,backprojection)
            output=self.network.activate(inputtrain)
            self._setLastOutput(output)
            outputs.append(self._out_layer.outputbuffer[self.network.offset - 1])

        return outputs

class FinancialTrainedNetwork(object):
    def __init__(self,stock):
        self.stock=stock
        self.network=NetworkReader.readFrom('trainednetworks/'+self.stock+'.xml')
        self.backprojectionFactor = 1


    def _getLastOutput(self):
        if self.network.offset == 0:
            return zeros(self.network.outdim)
        else:
            return self.network.outmodules[0].outputbuffer[self.network.offset - 1]

    def reset(self):
        self.network.reset()

    def activate(self,input):
        outputs=[]
        for val in input:
            backprojection=self._getLastOutput()
            backprojection*=self.backprojectionFactor
            inputtrain=append(val,backprojection)
            output=self.network.activate(inputtrain)
            self._setLastOutput(output)
            outputs.append(self.network.outmodules[0].outputbuffer[self.network.offset - 1])

        return outputs

    def _setLastOutput(self, output):
        self.network.outmodules[0].outputbuffer[self.network.offset - 1][:] = output