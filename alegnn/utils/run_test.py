#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import alegnn.utils.dataTools as dataTools
import alegnn.utils.graphML as gml
import alegnn.modules.architecturesTime as architTime
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation

#\\\ Separate functions:
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed, loadSeed


nAgents = 21

model_name = 'DAGNN1LyOptimInterval1600'

##creating directories
thisFilename = 'Test_SegregationGNN_'+model_name

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-%03d-' % nAgents + today



# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

saveDir = '/home/oyindamola/Research/graph-neural-networks/examples/experiments/SegregationGNN_-021-20220427010124DlAggGNN_K1_clipping__DAGGER_MAIn/'

k = 1

# Start measuring time
startRunTime = datetime.datetime.now()

#\\\ Basic parameters for the Aggregation GNN architecture

hParamsDAGNN1Ly = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

hParamsDAGNN1Ly['name'] = 'DAGNN1Ly'
# Chosen architecture
hParamsDAGNN1Ly['archit'] = architTime.AggregationGNN_DB
hParamsDAGNN1Ly['device'] = 'cuda:0' \
                                if (useGPU and torch.cuda.is_available()) \
                                else 'cpu'

# Graph convolutional parameters
hParamsDAGNN1Ly['dimFeatures'] = [10] # Features per layer
hParamsDAGNN1Ly['nFilterTaps'] = [] # Number of filter taps
hParamsDAGNN1Ly['bias'] = True # Decide whether to include a bias term
# Nonlinearity
hParamsDAGNN1Ly['nonlinearity'] = nonlinearity # Selected nonlinearity
    # is affected by the summary
hParamsDAGNN1Ly['poolingFunction'] = gml.NoPool
hParamsDAGNN1Ly['poolingSize'] = []
# Readout layer: local linear combination of features
hParamsDAGNN1Ly['dimReadout'] = [64, 2] # Dimension of the fully connected
    # layers after the GCN layers (map); this fully connected layer
    # is applied only at each node, without any further exchanges nor
    # considering all nodes at once, making the architecture entirely
    # local.
# Graph structure
hParamsDAGNN1Ly['dimEdgeFeatures'] = 1 # Scalar edge weights
hParamsDAGNN1Ly['nExchanges'] = k

# hParamsDAGNN1Ly['probExpert'] = probExpert

#\\\ Save Values:
# writeVarValues(varsFile, hParamsDAGNN1Ly)
thisModel = hParamsDAGNN1Ly['name']

########
# DATA #
########
clipping = True
alpha  = 2.0 # % control gain.
dAA    = 3.0  #% distance among same types.
dAB    = 6.0 # % distance among distinct types.
GROUPS = 3
useGPU = True # If true, and GPU is available, use it.
commRadius = 2 # Communication radius
duration = 25. # Duration of the trajectory
samplingTime = 0.05 # Sampling time
initGeometry = 'mine' # Geometry of initial positions
initVelValue = 0. # Initial velocities are samples from an interval
    # [-initVelValue, initVelValue]
initMinDist = 0.1 # No two agents are located at a distance less than this
accelMax = 10. # This is the maximum value of acceleration allowed

lossFunction = nn.MSELoss
thisLossFunction = lossFunction()
#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.0005 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

repelDist = .1 # Minimum distance before activating repelling potential
nTrain = 400 # Number of training samples
nValid = 1 # Number of valid samples
nTest = 20 # Number of testing samples

#\\\ Training algorithm
trainer = training.TrainerSegregate
thisTrainer = trainer

#\\\ Evaluation algorithm
evaluator = evaluation.evaluateSegregate
thisEvaluator = evaluator




thisOptim = optim.Adam(thisArchit.parameters(),
                       lr = learningRate,
                       betas = (beta1, beta2))



hParamsDict = deepcopy(eval('hParams' + thisModel))

callArchit = hParamsDict.pop('archit')
thisDevice = hParamsDict.pop('device')
thisName = hParamsDict.pop('name')

print("Initializing ARCHITECTURE")
thisArchit = callArchit(**hParamsDict)
thisArchit.to(thisDevice)


modelCreated = model.Model(thisArchit,
                           thisLossFunction,
                           thisOptim,
                           thisTrainer,
                           thisEvaluator,
                           thisDevice,
                           thisName,
                           saveDir)


modelCreated.load(label = 'Interval1600')



dataTest = dataTools.Segregation(
                # Structure
                nAgents,
                alpha,
                dAA, dAB, GROUPS,
                commRadius,
                0,
                clipping,
                # Samples
                1, # We don't care about training
                1, # nor validation
                nTest,
                # Time
                duration*2,
                samplingTime,
                # Initial conditions
                initGeometry = initGeometry,
                initVelValue = initVelValue,
                initMinDist = initMinDist,
                accelMax = accelMax)
