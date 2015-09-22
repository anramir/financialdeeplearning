__author__ = 'Andres'

from pybrain.supervised.evolino.population import EvolinoPopulation,EvolinoSubIndividual
from pybrain.supervised.trainers.trainer import Trainer
from pybrain.supervised.evolino.filter import EvolinoBurstMutation,EvolinoReproduction,EvolinoSelection,EvolinoSubMutation,EvolinoSubReproduction,EvolinoSubSelection
from pybrain.supervised.evolino.gfilter import Filter, SimpleMutation
from pybrain.supervised.evolino.gfilter import Randomization
from numpy import array, dot, concatenate, Infinity
from pybrain.tools.kwargsprocessor import KWArgsProcessor
from pybrain.tools.validation import Validator
from pybrain.supervised.evolino.variate import CauchyVariate
from scipy.linalg import pinv2
from copy import deepcopy

class EvolinoTrainer(Trainer):
    initialWeightRange = property(lambda self: self._initialWeightRange)
    subPopulationSize = property(lambda self: self._subPopulationSize)
    nCombinations = property(lambda self: self._nCombinations)
    nParents = property(lambda self: self._nParents)
    initialWeightRange = property(lambda self: self._initialWeightRange)
    mutationAlpha = property(lambda self: self._mutationAlpha)
    mutationVariate = property(lambda self: self._mutationVariate)
    wtRatio = property(lambda self: self._wtRatio)
    weightInitializer = property(lambda self: self._weightInitializer)
#    burstMutation        = property(lambda self: self._burstMutation)
    backprojectionFactor = property(lambda self: self._backprojectionFactor)
    def __init__(self,model,dataset,**kwargs):
        ap = KWArgsProcessor(self, kwargs)

        Trainer.__init__(self, model)

        ap = KWArgsProcessor(self, kwargs)

        # misc
        ap.add('verbosity', default=0)

        # population
        ap.add('subPopulationSize', private=True, default=8)
        ap.add('nCombinations', private=True, default=4)
        ap.add('nParents', private=True, default=None)
        ap.add('initialWeightRange', private=True, default=(-0.1, 0.1))
        ap.add('weightInitializer', private=True, default=Randomization(self._initialWeightRange[0], self._initialWeightRange[1]))

        # mutation
        ap.add('mutationAlpha', private=True, default=0.01)
        ap.add('mutationVariate', private=True, default=CauchyVariate(0, self._mutationAlpha))

        # evaluation
        ap.add('wtRatio', private=True, default=(1, 3))

        # burst mutation
        ap.add('nBurstMutationEpochs', default=Infinity)

        # network
        ap.add('backprojectionFactor', private=True, default=float(model.backprojectionFactor))
        model.backprojectionFactor = self._backprojectionFactor

        # aggregated objects
        ap.add('selection', default=EvolinoSelection())
        ap.add('reproduction', default=EvolinoReproduction(mutationVariate=self.mutationVariate))
        ap.add('burstMutation', default=EvolinoBurstMutation())
        ap.add('evaluation', default=EvolinoEvaluation(model, dataset, **kwargs))

        self.model=model
        self.dataset=dataset
        genome = self.model.getGenome()
        self.population = EvolinoPopulation(EvolinoSubIndividual(genome),self.subPopulationSize,self.nCombinations,self.weightInitializer)

        filters = []
        filters.append(self.evaluation)
        filters.append(self.selection)
        filters.append(self.reproduction)

        self._filters = filters

        self.totalepochs = 0
        self._max_fitness = self.evaluation.max_fitness
        self._max_fitness_epoch = self.totalepochs

    def train(self):
        """ Evolve for one epoch. """
        self.totalepochs += 1

        if self.totalepochs - self._max_fitness_epoch >= self.nBurstMutationEpochs:
            if self.verbosity: print("RUNNING BURST MUTATION")
            self.burstMutate()
            self._max_fitness_epoch = self.totalepochs


        for filter in self._filters:
            filter.apply(self.population)

        if self._max_fitness < self.evaluation.max_fitness:
            if self.verbosity: print(("GAINED FITNESS: ", self._max_fitness, " -->" , self.evaluation.max_fitness, "\n"))
            self._max_fitness = self.evaluation.max_fitness
            self._max_fitness_epoch = self.totalepochs
        else:
            if self.verbosity: print(("DIDN'T GAIN FITNESS:", "best =", self._max_fitness, "    current-best = ", self.evaluation.max_fitness, "\n"))

    def burstMutate(self):
        self.burstMutation.apply(self.population)


class EvolinoEvaluation(Filter):
    def __init__(self,model,dataset,**kwargs):

        Filter.__init__(self)
        ap = KWArgsProcessor(self, kwargs)
        ap.add('evalfunc', default=lambda output, target:-Validator.MSE(output, target))
        ap.add('verbosity', default=2)

        self.model=model
        self.dataset=dataset
        self.max_fitness=-Infinity


    def _evaluateNet(self):
        wtRatio=1./3.
        inputs=self.dataset.getField('input')
        targets=self.dataset.getField('target')

        training_start=int(wtRatio*len(inputs))
        washout_inputs=inputs[:training_start]
        training_inputs=inputs[training_start:]
        training_targets=targets[training_start:]
        phis=[]

        self.model.network.reset()

        self.model.washout(washout_inputs)
        phis.append(self.model.washout(training_inputs))

        PHI=concatenate(phis).T
        PHI_INV=pinv2(PHI)
        TARGET=concatenate(training_targets).T

        W=dot(TARGET,PHI_INV)
        self.model.setOutputWeightMatrix(W)

        self.model.activate(washout_inputs)
        outputs=self.model.activate(training_inputs)

        OUTPUT=concatenate(outputs)
        TARGET=TARGET.T

        fitness=self.evalfunc(OUTPUT,TARGET)

        return fitness


    def apply(self, population):
        population.clearFitness()
        best_W = None
        best_fitness = -Infinity

        for individual in population.getIndividuals():

            # load the individual's genome into the weights of the net
            self.model.setGenome(individual.getGenome())
            fitness = self._evaluateNet()
            if self.verbosity > 1:
                print(("Calculated fitness for individual", id(individual), " is ", fitness))

            # set the individual fitness
            population.setIndividualFitness(individual, fitness)

            if best_fitness < fitness:
                best_fitness = fitness
                best_genome = deepcopy(individual.getGenome())
                best_W = deepcopy(self.model.getOutputWeightMatrix())

        self.model.reset()
        self.model.setGenome(best_genome)
        self.model.setOutputWeightMatrix(best_W)


        # store fitness maximum to use it for triggering burst mutation
        self.max_fitness = best_fitness



