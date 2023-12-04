# 2020/01/01~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
# Kate Tolstaya, eig@seas.upenn.edu

# Learn decentralized controllers for flocking. There is a team of robots that
# start flying at random velocities and we want them to coordinate so that they
# can fly together while avoiding collisions. We learn a decentralized
# controller by using imitation learning.

# In this simulation, the number of agents is fixed for training, but can be
# set to a different number for testing.

# Outputs:
# - Text file with all the hyperparameters selected for the run and the
#   corresponding results (hyperparameters.txt)
# - Pickle file with the random seeds of both torch and numpy for accurate
#   reproduction of results (randomSeedUsed.pkl)
# - The parameters of the trained models, for both the Best and the Last
#   instance of each model (savedModels/)
# - The figures of loss and evaluation through the training iterations for
#   each model (figs/ and trainVars/)
# - Videos for some of the trajectories in the dataset, following the optimal
#   centralized controller (datasetTrajectories/)
# - Videos for some of the learned trajectories following the controles
#   learned by each model (learnedTrajectories/)

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

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
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
import gc
# del variables
gc.collect()
#\\\ Own libraries:
import alegnn.utils.dataTools as dataTools
import alegnn.utils.graphML as gml
import alegnn.modules.architecturesTime as architTime
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation
from alegnn.modules.logger import Logger
#\\\ Separate functions:
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed, loadSeed

# Start measuring time
startRunTime = datetime.datetime.now()
from collections import OrderedDict
#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

# Select desired architectures
doLocalFlt = False # Local filter (no nonlinearity)
doLocalGNN = False # Local GNN (include nonlinearity)
doDlAggGNN = True
doGraphRNN = False

reload = True
create_folder = True
lodDir ='SegregationGNN_NStates14_-021-20230204174615DlAggGNN_K3_clipping__DAGGER_'
label_name = 'Interval2200'
startEpoch = 0

f_name = ''
k=0

# groups_ = ['2', '5', '10', '5']
# agents = ['50', '50', '50','100']
# groups_ = ['2']
# agents = ['10']

# groups_ = ['2','5', '3' , '7', '5', '2', '5', '10', '5']
# agents = ['10','20', '21', '21', '30', '50', '50', '50','100']
groups_ = [ '10', '5']
agents = [ '50','100']


for ik in range(1):
	# if ik ==0:
	# 	dAA    = 5.0  #% distance among same types.
	# 	dAB    = 10.0 # % distance among distinct types.
	# 	exp ="Segregation"
	# else:
	dAA    = 5.0  #% distance among same types.
	dAB    = 10 # % distance among distinct types.
	exp ="Segregation"
	for i in range(len(agents)):
		clipping = True
		k =3

		dagger = True
		# print("K", k)
		if doLocalFlt:
			f_name = 'LocalFit'
		if doLocalGNN:
			f_name = 'LocalGNN'
		if doDlAggGNN:
			#k = 4
			f_name = 'DlAggGNN_K'+str(k)
		if doGraphRNN:
			f_name = 'doGraphRNN'
		if clipping:
			f_name = f_name+'_clipping_'
		else:
			f_name = f_name+'_noclipping_'

		if dagger:
			f_name = f_name+'_DAGGER_'
		else:
			f_name = f_name+'_noDAGGER_'

		nAgents = int(agents[i]) # Number of agents at training time
		GROUPS = int(groups_[i])
		nstates = 14


		thisFilename = 'SegregationGNN_NStates'+str(nstates)+ '_' # This is the general name of all related files



		saveDirRoot = 'experiments' # In this case, relative location
		saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
			# the results from each run

		#\\\ Create .txt to store the values of the setting parameters for easier
		# reference when running multiple experiments
		today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
		# Append date and time of the run to the directory, to avoid several runs of
		# overwritting each other.
		saveDir = saveDir + '-%03d-' % nAgents + today + f_name
		saveDir_ = os.path.join(saveDirRoot, lodDir)


		if reload and create_folder:
			saveDir = os.path.join(saveDirRoot, lodDir)
			startEpoch = 300
		# print(saveDir)
		# Create directory
		if not os.path.exists(saveDir):
			os.makedirs(saveDir)

		varsFile = os.path.join(saveDir,'hyperparameters.txt')
		if not reload:
			# Create the file where all the (hyper)parameters and results will be saved.
			with open(varsFile, 'w+') as file:
				file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

		# \\\ Save seeds for reproducibility
		  # PyTorch seeds

		torchState = torch.get_rng_state()
		torchSeed = torch.initial_seed()
		#   Numpy seeds
		numpyState = np.random.RandomState().get_state()
		#   Collect all random states
		# print(numpyState, numpyState, numpyState)
		randomStates = []
		randomStates.append({})
		randomStates[0]['module'] = 'numpy'
		randomStates[0]['state'] = numpyState
		randomStates.append({})
		randomStates[1]['module'] = 'torch'
		randomStates[1]['state'] = numpyState
		randomStates[1]['seed'] = numpyState
		  # This list and dictionary follows the format to then be loaded, if needed,
		  # by calling the loadSeed function in Utils.miscTools
		saveSeed(randomStates, saveDir)
		# loadSeed('/home/oyindamola/Research/graph-neural-networks/examples/experiments/SegregationGNN_NStates12_-021-20220510154928DlAggGNN_K3_clipping__DAGGER_')

		########
		# DATA #
		########
		alpha  = 3.0 # % control gain.


		useGPU = True # If true, and GPU is available, use it.

		nAgentsMax = nAgents # Maximum number of agents to test the solution
		nSimPoints = 1 # Number of simulations between nAgents and nAgentsMax
			# At test time, the architectures trained on nAgents will be tested on a
			# varying number of agents, starting at nAgents all the way to nAgentsMax;
			# the number of simulations for different number of agents is given by
			# nSimPoints, i.e. if nAgents = 50, nAgentsMax = 100 and nSimPoints = 3,
			# then the architectures are trained on 50, 75 and 100 agents.
		commRadius = 12 # Communication radius
		repelDist = .1 # Minimum distance before activating repelling potential
		nTrain = 1 # Number of training samples
		nValid = 1 # Number of valid samples
		nTest = 40 # Number of testing samples

		duration = 300. # Duration of the trajectory
		samplingTime = 0.05 # Sampling time
		initGeometry = 'mine' # Geometry of initial positions
		initVelValue = 0. # Initial velocities are samples from an interval
			# [-initVelValue, initVelValue]
		initMinDist = 0.1 # No two agents are located at a distance less than this
		accelMax = 1. # This is the maximum value of acceleration allowed

		nRealizations = 1 # Number of data realizations
			# How many times we repeat the experiment

		if not reload:
			#\\\ Save values:
			writeVarValues(varsFile,
						   {'nAgents': nAgents,
						   'dAA': dAA,
						   'dAB': dAB,
						   'alpha': alpha,
						   'GROUPS': GROUPS,
							'nAgentsMax': nAgentsMax,
							'nSimPoints': nSimPoints,
							'commRadius': commRadius,
							'repelDist': repelDist,
							'nTrain': nTrain,
							'nValid': nValid,
							'nTest': nTest,
							'clipping':clipping,
							'duration': duration,
							'samplingTime': samplingTime,
							'initGeometry': initGeometry,
							'initVelValue': initVelValue,
							'initMinDist': initMinDist,
							'accelMax': accelMax,
							'nRealizations': nRealizations,
							'useGPU': useGPU})

		############
		# TRAINING #
		############

		#\\\ Individual model training options
		optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
		learningRate = 0.0005 # In all options
		beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
		beta2 = 0.999 # ADAM option only

		#\\\ Loss function choice
		lossFunction = nn.MSELoss

		#\\\ Training algorithm
		trainer = training.TrainerSegregate

		#\\\ Evaluation algorithm
		evaluator = evaluation.evaluateSegregate

		#\\\ Overall training options
		probExpert = 0.993 # Probability of choosing the expert in DAGger
		# DAGgerType = 'fixedBatch' # 'replaceTimeBatch', 'randomEpoch'
		DAGgerType =  'randomEpoch'
		nEpochs = 4000 # Number of epochs
		batchSize = 200 # Batch size
		doLearningRateDecay = False # Learning rate decay
		learningRateDecayRate = 0.9 # Rate
		learningRateDecayPeriod = 100 # How many epochs after which update the lr
		validationInterval = 20 # How many training steps to do the validation

		#\\\ Save values
		if not reload:
			writeVarValues(varsFile,
						   {'optimizationAlgorithm': optimAlg,
							'learningRate': learningRate,
							'beta1': beta1,
							'beta2': beta2,
							'lossFunction': lossFunction,
							'trainer': trainer,
							'evaluator': evaluator,
							'probExpert': probExpert,
							'nEpochs': nEpochs,
							'batchSize': batchSize,
							'doLearningRateDecay': doLearningRateDecay,
							'learningRateDecayRate': learningRateDecayRate,
							'learningRateDecayPeriod': learningRateDecayPeriod,
							'validationInterval': validationInterval})

		#################
		# ARCHITECTURES #
		#################

		# In this section, we determine the (hyper)parameters of models that we are
		# going to train. This only sets the parameters. The architectures need to be
		# created later below. Do not forget to add the name of the architecture
		# to modelList.

		# If the hyperparameter dictionary is called 'hParams' + name, then it can be
		# picked up immediately later on, and there's no need to recode anything after
		# the section 'Setup' (except for setting the number of nodes in the 'N'
		# variable after it has been coded).

		# The name of the keys in the hyperparameter dictionary have to be the same
		# as the names of the variables in the architecture call, because they will
		# be called by unpacking the dictionary.

		#nFeatures = 32 # Number of features in all architectures
		#nFilterTaps = 4 # Number of filter taps in all architectures
		# [[The hyperparameters are for each architecture, and they were chosen
		#   following the results of the hyperparameter search]]
		nonlinearityHidden = torch.tanh
		nonlinearityOutput = torch.tanh
		nonlinearity = nn.Tanh # Chosen nonlinearity for nonlinear architectures



		modelList = []

		#\\\\\\\\\\\\\\\\\\
		#\\\ FIR FILTER \\\
		#\\\\\\\\\\\\\\\\\\

		if doLocalFlt:

			#\\\ Basic parameters for the Local Filter architecture

			hParamsLocalFlt = {} # Hyperparameters (hParams) for the Local Filter

			hParamsLocalFlt['name'] = 'LocalFlt'
			# Chosen architecture
			hParamsLocalFlt['archit'] = architTime.LocalGNN_DB
			hParamsLocalFlt['device'] = 'cuda:0' \
											if (useGPU and torch.cuda.is_available()) \
											else 'cpu'

			# Graph convolutional parameters
			hParamsLocalFlt['dimNodeSignals'] = [10, 32] # Features per layer
			hParamsLocalFlt['nFilterTaps'] = [4] # Number of filter taps
			hParamsLocalFlt['bias'] = True # Decide whether to include a bias term
			# Nonlinearity
			hParamsLocalFlt['nonlinearity'] = gml.NoActivation # Selected nonlinearity
				# is affected by the summary
			# Readout layer: local linear combination of features
			hParamsLocalFlt['dimReadout'] = [2] # Dimension of the fully connected
				# layers after the FIR filter layers (map); this fully connected layer
				# is applied only at each node, without any further exchanges nor
				# considering all nodes at once, making the architecture entirely
				# local.
			# Graph structure
			hParamsLocalFlt['dimEdgeFeatures'] = 1 # Scalar edge weights

			#\\\ Save Values:
			writeVarValues(varsFile, hParamsLocalFlt)
			modelList += [hParamsLocalFlt['name']]

		#\\\\\\\\\\\\\\\\\
		#\\\ LOCAL GNN \\\
		#\\\\\\\\\\\\\\\\\

		if doLocalGNN:

			#\\\ Basic parameters for the Local GNN architecture

			hParamsLocalGNN = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

			hParamsLocalGNN['name'] = 'LocalGNN'
			# Chosen architecture
			hParamsLocalGNN['archit'] = architTime.LocalGNN_DB
			hParamsLocalGNN['device'] = 'cuda:0' \
											if (useGPU and torch.cuda.is_available()) \
											else 'cpu'

			# Graph convolutional parameters
			hParamsLocalGNN['dimNodeSignals'] = [10, 64] # Features per layer
			hParamsLocalGNN['nFilterTaps'] = [3] # Number of filter taps
			hParamsLocalGNN['bias'] = True # Decide whether to include a bias term
			# Nonlinearity
			hParamsLocalGNN['nonlinearity'] = nonlinearity # Selected nonlinearity
				# is affected by the summary
			# Readout layer: local linear combination of features
			hParamsLocalGNN['dimReadout'] = [2] # Dimension of the fully connected
				# layers after the GCN layers (map); this fully connected layer
				# is applied only at each node, without any further exchanges nor
				# considering all nodes at once, making the architecture entirely
				# local.
			# Graph structure
			hParamsLocalGNN['dimEdgeFeatures'] = 1 # Scalar edge weights

			#\\\ Save Values:
			writeVarValues(varsFile, hParamsLocalGNN)
			modelList += [hParamsLocalGNN['name']]

		#\\\\\\\\\\\\\\\\\\\\\\\
		#\\\ AGGREGATION GNN \\\
		#\\\\\\\\\\\\\\\\\\\\\\\

		if doDlAggGNN:

			#\\\ Basic parameters for the Aggregation GNN architecture

			hParamsDAGNN1Ly = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

			hParamsDAGNN1Ly['name'] = 'DAGNN1Ly'
			# Chosen architecture
			hParamsDAGNN1Ly['archit'] = architTime.AggregationGNN_DB
			hParamsDAGNN1Ly['device'] = 'cuda:0' \
											if (useGPU and torch.cuda.is_available()) \
											else 'cpu'

			# Graph convolutional parameters
			hParamsDAGNN1Ly['dimFeatures'] = [nstates] # Features per layer
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
			writeVarValues(varsFile, hParamsDAGNN1Ly)
			modelList += [hParamsDAGNN1Ly['name']]

		#\\\\\\\\\\\\\\\\\
		#\\\ GRAPH RNN \\\
		#\\\\\\\\\\\\\\\\\

		if doGraphRNN:

			#\\\ Basic parameters for the Graph RNN architecture

			hParamsGraphRNN = {} # Hyperparameters (hParams) for the Local GNN (LclGNN)

			hParamsGraphRNN['name'] = 'GraphRNN'
			# Chosen architecture
			hParamsGraphRNN['archit'] = architTime.GraphRecurrentNN_DB
			hParamsGraphRNN['device'] = 'cuda:0' \
											if (useGPU and torch.cuda.is_available()) \
											else 'cpu'

			# Graph convolutional parameters
			hParamsGraphRNN['dimInputSignals'] = nstates # Features per layer
			hParamsGraphRNN['dimOutputSignals'] = 64
			hParamsGraphRNN['dimHiddenSignals'] = 64
			hParamsGraphRNN['nFilterTaps'] = [3] * 2 # Number of filter taps
			hParamsGraphRNN['bias'] = True # Decide whether to include a bias term
			# Nonlinearity
			hParamsGraphRNN['nonlinearityHidden'] = nonlinearityHidden
			hParamsGraphRNN['nonlinearityOutput'] = nonlinearityOutput
			hParamsGraphRNN['nonlinearityReadout'] = nonlinearity
			# Readout layer: local linear combination of features
			hParamsGraphRNN['dimReadout'] = [2] # Dimension of the fully connected
				# layers after the GCN layers (map); this fully connected layer
				# is applied only at each node, without any further exchanges nor
				# considering all nodes at once, making the architecture entirely
				# local.
			# Graph structure
			hParamsGraphRNN['dimEdgeFeatures'] = 1 # Scalar edge weights
			# hParamsGraphRNN['probExpert'] = probExpert
			#\\\ Save Values:
			writeVarValues(varsFile, hParamsGraphRNN)
			modelList += [hParamsGraphRNN['name']]

		###########
		# LOGGING #
		###########

		# Options:
		doPrint = True # Decide whether to print stuff while running
		doLogging = False # Log into tensorboard
		doSaveVars = True # Save (pickle) useful variables
		doFigs = True # Plot some figures (this only works if doSaveVars is True)
		# Parameters:
		printInterval = 1 # After how many training steps, print the partial results
		#   0 means to never print partial results while training
		xAxisMultiplierTrain = 10 # How many training steps in between those shown in
			# the plot, i.e., one training step every xAxisMultiplierTrain is shown.
		xAxisMultiplierValid = 2 # How many validation steps in between those shown,
			# same as above.
		figSize = 5 # Overall size of the figure that contains the plot
		lineWidth = 2 # Width of the plot lines
		markerShape = 'o' # Shape of the markers
		markerSize = 3 # Size of the markers
		videoSpeed = 5 #0.5 # Slow down by half to show transitions
		nVideos = 1 # Number of videos to save
		# print("saveDir:", saveDir)

		#\\\ Save values:
		writeVarValues(varsFile,
					   {'doPrint': doPrint,
						'doLogging': doLogging,
						'doSaveVars': doSaveVars,
						'doFigs': doFigs,
						'saveDir': saveDir,
						'printInterval': printInterval,
						'figSize': figSize,
						'lineWidth': lineWidth,
						'markerShape': markerShape,
						'markerSize': markerSize,
						'videoSpeed': videoSpeed,
						'nVideos': nVideos})

		#%%##################################################################
		#                                                                   #
		#                    SETUP                                          #
		#                                                                   #
		#####################################################################

		#\\\ If CUDA is selected, empty cache:
		if useGPU and torch.cuda.is_available():
			torch.cuda.empty_cache()

		print("modelList: ", modelList)
		#\\\ Notify of processing units
		if doPrint:
			print("Selected devices:")
			for thisModel in modelList:
				hParamsDict = eval('hParams' + thisModel)
				print(hParamsDict)
				print("\t%s: %s" % (thisModel, hParamsDict['device']))

		#\\\ Logging options
		if doLogging:
			# If logging is on, load the tensorboard visualizer and initialize it
			from alegnn.utils.visualTools import Visualizer
			logsTB = os.path.join(saveDir, 'logsTB')
			logger = Visualizer(logsTB, name='visualResults')

		#\\\ Number of agents at test time
		nAgentsTest = np.linspace(nAgents, nAgentsMax, num = nSimPoints,dtype = np.int)
		nAgentsTest = np.unique(nAgentsTest).tolist()
		nSimPoints = len(nAgentsTest)
		writeVarValues(varsFile, {'nAgentsTest': nAgentsTest}) # Save list

		#\\\ Save variables during evaluation.
		# We will save all the evaluations obtained for each of the trained models.
		# The first list is one for each value of nAgents that we want to simulate
		# (i.e. these are test results, so if we test for different number of agents,
		# we need to save the results for each of them). Each element in the list will
		# be a dictionary (i.e. for each testing case, we have a dictionary).
		# It basically is a dictionary, containing a list. The key of the
		# dictionary determines the model, then the first list index determines
		# which split realization. Then, this will be converted to numpy to compute
		# mean and standard deviation (across the split dimension).
		# We're saving the cost of the full trajectory, as well as the cost at the end
		# instant.
		costBestFull = [None] * nSimPoints
		costBestEnd = [None] * nSimPoints
		costLastFull = [None] * nSimPoints
		costLastEnd = [None] * nSimPoints
		costOptFull = [None] * nSimPoints
		costOptEnd = [None] * nSimPoints
		for n in range(nSimPoints):
			costBestFull[n] = {} # Accuracy for the best model (full trajectory)
			costBestEnd[n] = {} # Accuracy for the best model (end time)
			costLastFull[n] = {} # Accuracy for the last model
			costLastEnd[n] = {} # Accuracy for the last model
			for thisModel in modelList: # Create an element for each split realization,
				costBestFull[n][thisModel] = [None] * nRealizations
				costBestEnd[n][thisModel] = [None] * nRealizations
				costLastFull[n][thisModel] = [None] * nRealizations
				costLastEnd[n][thisModel] = [None] * nRealizations
			costOptFull[n] = [None] * nRealizations # Accuracy for optimal controller
			costOptEnd[n] = [None] * nRealizations # Accuracy for optimal controller

		if doFigs:
			#\\\ SAVE SPACE:
			# Create the variables to save all the realizations. This is, again, a
			# dictionary, where each key represents a model, and each model is a list
			# for each data split.
			# Each data split, in this case, is not a scalar, but a vector of
			# length the number of training steps (or of validation steps)
			lossTrain = {}
			evalValid = {}
			# Initialize the splits dimension
			for thisModel in modelList:
				lossTrain[thisModel] = [None] * nRealizations
				evalValid[thisModel] = [None] * nRealizations
		#
		#
		####################
		# TRAINING OPTIONS #
		####################

		# Training phase. It has a lot of options that are input through a
		# dictionary of arguments.
		# The value of these options was decided above with the rest of the parameters.
		# This just creates a dictionary necessary to pass to the train function.

		trainingOptions = {}
		print("saveDir:", saveDir)
		#
		if doLogging:
			trainingOptions['logger'] = logger
		if doSaveVars:
			trainingOptions['saveDir'] = saveDir
		if doPrint:
			trainingOptions['printInterval'] = printInterval
		if doLearningRateDecay:
			trainingOptions['learningRateDecayRate'] = learningRateDecayRate
			trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
		trainingOptions['validationInterval'] = validationInterval

		# And in case each model has specific training options (aka 'DAGger'), then
		# we create a separate dictionary per model.
		#
		trainingOptsPerModel= {}
		#
		# Create relevant dirs: we need directories to save the videos of the dataset
		# that involve the optimal centralized controllers, and we also need videos
		# for the learned trajectory of each model. Note that all of these depend on
		# each realization, so we will be saving videos for each realization.
		# Here, we create all those directories.
		datasetTrajectoryDir = os.path.join(saveDir,'datasetTrajectories')
		if not os.path.exists(datasetTrajectoryDir):
			os.makedirs(datasetTrajectoryDir)

		datasetTrainTrajectoryDir = os.path.join(datasetTrajectoryDir,'train')
		if not os.path.exists(datasetTrainTrajectoryDir):
			os.makedirs(datasetTrainTrajectoryDir)

		datasetTestTrajectoryDir = os.path.join(datasetTrajectoryDir,'test')
		if not os.path.exists(datasetTestTrajectoryDir):
			os.makedirs(datasetTestTrajectoryDir)

		datasetTestAgentTrajectoryDir = [None] * nSimPoints
		for n in range(nSimPoints):
			datasetTestAgentTrajectoryDir[n] = os.path.join(datasetTestTrajectoryDir,
															'%03d' % nAgentsTest[n])

		if nRealizations > 1:
			datasetTrainTrajectoryDirOrig = datasetTrainTrajectoryDir
			datasetTestAgentTrajectoryDirOrig = datasetTestAgentTrajectoryDir.copy()

		# #%%##################################################################
		# #                                                                   #
		# #                    DATA SPLIT REALIZATION                         #
		# #                                                                   #
		# #####################################################################

		# Start generating a new data realization for each number of total realizations

		for realization in range(nRealizations):

			# On top of the rest of the training options, we pass the identification
			# of this specific data split realization.

			if nRealizations > 1:
				trainingOptions['realizationNo'] = realization

				# Create new directories (specific for this realization)
				datasetTrainTrajectoryDir = os.path.join(datasetTrainTrajectoryDirOrig,
														 '%03d' % realization)
				if not os.path.exists(datasetTrainTrajectoryDir):
					os.makedirs(datasetTrainTrajectoryDir)

				for n in range(nSimPoints):
					datasetTestAgentTrajectoryDir[n] = os.path.join(
												  datasetTestAgentTrajectoryDirOrig[n],
												  '%03d' % realization)
					if not os.path.exists(datasetTestAgentTrajectoryDir[n]):
						os.makedirs(datasetTestAgentTrajectoryDir[n])

			if doPrint:
				print("", flush = True)

			#%%##################################################################
			#                                                                   #
			#                    DATA HANDLING                                  #
			#                                                                   #
			#####################################################################

			############
			# DATASETS #
			############

			if doPrint:
				print("Generating data", end = '')
				if nRealizations > 1:
					print(" for realization %d" % realization, end = '')
				print("...", flush = True)

			initPos = np.load('/home/oyindamola/Research/Current_GNN_Research/graph-neural-networks/alegnn/utils/Testsegregate_initPos_Agent'+str(nAgents)+'_GR'+str(GROUPS)+'_CR'+str(commRadius)+'_Samples42.npy',allow_pickle=True)
			initVel = np.load('/home/oyindamola/Research/Current_GNN_Research/graph-neural-networks/alegnn/utils/Testsegregate_initVel_Agent'+str(nAgents)+'_GR'+str(GROUPS)+'_CR'+str(commRadius)+'_Samples42.npy',allow_pickle=True)
			initData = (initPos, initVel)
			print("init: ", len(initData))
			#   Generate the dataset
			data = dataTools.Segregation(
						# Structure
						nAgents,
						nstates,
						alpha,
						dAA, dAB, GROUPS,
						commRadius,
						repelDist,
						clipping,
						# Samples
						1,
						1,
						40, # We do not care about testing, we will re-generate the
						   # dataset for testing
						# Time
						duration,
						samplingTime,
						# Initial conditions
						initGeometry = initGeometry,
						initVelValue = initVelValue,
						initMinDist = initMinDist,
						accelMax = accelMax,
						normalizeGraph =False,
						data_=initData)

			###########
			# PREVIEW #
			###########

			if doPrint:
				print("Preview data", end = '')
				if nRealizations > 1:
					print(" for realization %d" % realization, end = '')
				print("...", flush = True)

			# Generate the videos
			# data.saveVideo(datasetTrainTrajectoryDir, # Where to save them
			#                 data.pos['train'], # Which positions to plot
			#                 nVideos, # Number of videos to create
			#                 commGraph = data.commGraph['train'], # Graph to plot
			#                 vel = data.vel['train'], # Velocity arrows to plot
			#                 videoSpeed = videoSpeed) # Change speed of animation

			# Generate the videos
			# data.saveVideo(datasetTestTrajectoryDir, # Where to save them
			#                 data.local_pos['train'], # Which positions to plot
			#                 nVideos, # Number of videos to create
			#                 commGraph = data.commGraph['train'], # Graph to plot
			#                 vel = data.local_vel['train'], # Velocity arrows to plot
			#                 videoSpeed = videoSpeed) # Change speed of animation

			# data.saveVideo(datasetTestTrajectoryDir, # Where to save them
			#                 data.pos['valid'], # Which positions to plot
			#                 nVideos, # Number of videos to create
			#                 commGraph = data.commGraph['valid'], # Graph to plot
			#                 vel = data.vel['valid'], # Velocity arrows to plot
			#                 videoSpeed = videoSpeed) # Change speed of animation

			#%%##################################################################
			#                                                                   #
			#                    MODELS INITIALIZATION                          #
			#                                                                   #
			#####################################################################

			# This is the dictionary where we store the models (in a model.Model
			# class).
			modelsGNN = {}

			# If a new model is to be created, it should be called for here.

			if doPrint:
				print("Model initialization...", flush = True)

			for thisModel in modelList:

				# Get the corresponding parameter dictionary
				hParamsDict = deepcopy(eval('hParams' + thisModel))
				if dagger:
					hParamsDict['DAGgerType'] = DAGgerType
					hParamsDict['probExpert'] =probExpert
				# print("Here", trainingOptions)
				# and training options
				trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)

				# print("Here", hParamsDict)

				# Now, this dictionary has all the hyperparameters that we need to pass
				# to the architecture, but it also has the 'name' and 'archit' that
				# we do not need to pass them. So we are going to get them out of
				# the dictionary
				thisName = hParamsDict.pop('name')
				callArchit = hParamsDict.pop('archit')
				thisDevice = hParamsDict.pop('device')
				# If there's a specific DAGger type, pop it out now
				# print(hParamsDict.keys() )
				if 'DAGgerType' in hParamsDict.keys() \
												and 'probExpert' in hParamsDict.keys():
					trainingOptsPerModel[thisModel]['probExpert'] = \
														  hParamsDict.pop('probExpert')
					trainingOptsPerModel[thisModel]['DAGgerType'] = \
														  hParamsDict.pop('DAGgerType')
					print("DAGgertpe:", DAGgerType)

				# If more than one graph or data realization is going to be carried out,
				# we are going to store all of thos models separately, so that any of
				# them can be brought back and studied in detail.
				if nRealizations > 1:
					thisName += 'G%02d' % realization

				if doPrint:
					print("\tInitializing %s..." % thisName,
						  end = ' ',flush = True)

				##############
				# PARAMETERS #
				##############

				#\\\ Optimizer options
				#   (If different from the default ones, change here.)
				thisOptimAlg = optimAlg
				thisLearningRate = learningRate
				thisBeta1 = beta1
				thisBeta2 = beta2

				################
				# ARCHITECTURE #
				################

				print("Initializing ARCHITECTURE")
				thisArchit = callArchit(**hParamsDict)
				thisArchit.to(thisDevice)

				#############
				# OPTIMIZER #
				#############

				if thisOptimAlg == 'ADAM':
					thisOptim = optim.Adam(thisArchit.parameters(),
										   lr = learningRate,
										   betas = (beta1, beta2))
				elif thisOptimAlg == 'SGD':
					thisOptim = optim.SGD(thisArchit.parameters(),
										  lr = learningRate)
				elif thisOptimAlg == 'RMSprop':
					thisOptim = optim.RMSprop(thisArchit.parameters(),
											  lr = learningRate, alpha = beta1)

				########
				# LOSS #
				########

				thisLossFunction = lossFunction()

				###########
				# TRAINER #
				###########

				thisTrainer = trainer

				#############
				# EVALUATOR #
				#############

				thisEvaluator = evaluator

				#########
				# MODEL #
				#########

				modelCreated = model.Model(thisArchit,
										   thisLossFunction,
										   thisOptim,
										   thisTrainer,
										   thisEvaluator,
										   thisDevice,
										   thisName,
										   saveDir)


				if reload:
					modelCreated.load(save_dir = saveDir_, label = label_name)

				modelsGNN[thisName] = modelCreated

				writeVarValues(varsFile,
							   {'name': thisName,
								'thisOptimizationAlgorithm': thisOptimAlg,
								'thisTrainer': thisTrainer,
								'thisEvaluator': thisEvaluator,
								'thisLearningRate': thisLearningRate,
								'thisBeta1': thisBeta1,
								'thisBeta2': thisBeta2})

				if doPrint:
					print("OK")

			#%%##################################################################
			#                                                                   #
			#                    TRAINING                                       #
			#                                                                   #
			#####################################################################


			# # ############
			# # # TRAINING #
			# # ############
			# #
			# print("")
			# print("modelsGNN.keys(): ", modelsGNN.keys())
			# for thisModel in modelsGNN.keys():
			#     print("thisModel: ", thisModel)
			#
			#     if doPrint:
			#         print("Training model %s..." % thisModel)
			#
			#     for m in modelList:
			#         if m in thisModel:
			#             modelName = m
			#
			#     thisTrainVars = modelsGNN[thisModel].train(data,
			#                                                nEpochs,
			#                                                startEpoch,
			#                                                batchSize,
			#                                                **trainingOptsPerModel[m])
			#
			#     if doFigs:
			#     # Find which model to save the results (when having multiple
			#     # realizations)
			#         for m in modelList:
			#             if m in thisModel:
			#                 lossTrain[m][realization] = thisTrainVars['lossTrain']
			#                 evalValid[m][realization] = thisTrainVars['evalValid']
			# # And we also need to save 'nBatch' but is the same for all models, so
			# if doFigs:
			#     nBatches = thisTrainVars['nBatches']

			#%%##################################################################
			#                                                                   #
			#                    EVALUATION                                     #
			#                                                                   #
			#####################################################################

			# Now that the model has been trained, we evaluate them on the test
			# samples.

			# We have two versions of each model to evaluate: the one obtained
			# at the best result of the validation step, and the last trained model.

			initPosValid = data.getData('initPos','test')
			initVelValid = data.getData('initVel','test')



			print("initPosValid:", initPosValid.shape)

			#
			#
			# # Compute trajectories
			# # posTestValid, velTestValid, _, _, graphTestValid = data.computeTrajectory(
			# # 		initPosValid_, initVelValid_, data.duration,
			# # 		archit = thisArchit, doPrint = False, clipping=True)
			#
			# gl__ =[ 'True', 'False']
			#
			# for gh in range(len(gl__)):
			# 	gl_ = gl__[gh]
			#
			# 	if gl_=='True':
			# 		type_alg = 'global'
			# 		posTestValid = np.load('./GlobalPos_'+str(nAgents)+'_'+str(GROUPS)+'_'+str(alpha)+'.npy')[0:nTest]
			# 		velTestValid = np.load('./GlobalVel_'+str(nAgents)+'_'+str(GROUPS)+'_'+str(alpha)+'.npy')[0:nTest]
			#
			# 		# posTestValid = data.posAll[0:nTest]
			# 		# velTestValid = data.velAll[0:nTest]
			# 	elif gl_ =='learner':
			# 		type_alg = 'learner'
			# 		posTestValid, velTestValid, _, _, graphTestValid = data.computeTrajectory(
			# 				initPosValid_, initVelValid_, data.duration,
			# 				archit = thisArchit, doPrint = False, clipping=True)
			#
			# 	else:
			# 		type_alg = 'local'
			# 		# posTestValid = data.localposAll[0:nTest]
			# 		# velTestValid = data.localvelAll[0:nTest]
			# 		posTestValid = np.load('./LocalPos_'+str(nAgents)+'_'+str(GROUPS)+'_'+str(alpha)+'.npy')[0:nTest]
			# 		velTestValid = np.load('./LocalVel_'+str(nAgents)+'_'+str(GROUPS)+'_'+str(alpha)+'.npy')[0:nTest]
			#
			#
			#
			# 	t_len = posTestValid.shape[0]
			#
			#
			#
			#
			# 	# posTestValid, velTestValid, _ = data.computeOptimalTrajectory(
			# 	# 		initPosValid, initVelValid, data.duration,
			# 	# 		samplingTime, repelDist, alpha, data.const,data.accelMax)
			#
			# 	print("g: ", posTestValid.shape)
			#
			# 	# log_dir = modelCreated.saveDir+"/"+str(nAgents)+"_"+str(GROUPS)+"_"+today+"/"+str(i)
			# 	#
			# 	# data.saveVideo(log_dir, # Where to save them
			# 	# 				posTestValid, # Which positions to plot
			# 	# 				1, # Number of videos to create
			# 	# 				commGraph = graphTestValid, # Graph to plot
			# 	# 				vel = velTestValid, # Velocity arrows to plot
			# 	# 				videoSpeed = 5,
			# 	# 				videoNum= "x_1900",if_gif=False) # Change speed of animation
			# 	#
			#
			#
			#
			# 	for ii in range(t_len):
			#
			# 		log_dir = modelCreated.saveDir+"/"+type_alg+"/alpha"+str(alpha)+'_'+str(nAgents)+"_"+str(GROUPS)+"_"+exp+'_'+today+"/"+str(ii)
			# 		posTestValid_ = np.expand_dims(posTestValid[ii],0)
			# 		velTestValid_ = np.expand_dims(velTestValid[ii],0)
			#
			# 		print("ii", ii, t_len-1)
			#
			#
			# 		if not os.path.exists(log_dir):
			# 			os.makedirs(log_dir)
			# 		videoName = 'sampleTrajectory.gif'
			# 		saveDir = os.path.join(log_dir, videoName)
			#
			# 		if ii == 0:
			# 			data.make_gif(posTestValid_[0], saveDir,fps=60)
			#
			# 		logger_ = Logger(log_dir)
			#
			#
			# 		stats_ = data.evaluate_all(pos=posTestValid_, \
			# 		vel = velTestValid_)
			#
			#
			#
			# 		for jj in range(len(stats_)):
			# 			logs = OrderedDict()
			# 			data.perform_logging(jj, logs, logger_, stats_[jj], False)
			#
			#


				# for i in range(t_len):
				# 	initPosValid_ = np.expand_dims(initPosValid[i],0)
				# 	initVelValid_ = np.expand_dims(initVelValid[i],0)
				# 	# Compute trajectories
				# 	# posTestValid, velTestValid, _, _, graphTestValid = data.computeTrajectory(
				# 	# 		initPosValid_, initVelValid_, data.duration,
				# 	# 		archit = thisArchit, doPrint = False, clipping=True)
				#
				# 	posTestValid, velTestValid, _ = data.computeOptimalTrajectory(
				# 			initPosValid_, initVelValid_, data.duration,
				# 			samplingTime, repelDist, alpha, data.const,data.accelMax)
				#
				# 	log_dir = modelCreated.saveDir+"/"+str(nAgents)+"_"+str(GROUPS)+"_"+today+"/"+str(i)
				# 	#
				# 	# data.saveVideo(log_dir, # Where to save them
				# 	# 				posTestValid, # Which positions to plot
				# 	# 				1, # Number of videos to create
				# 	# 				commGraph = graphTestValid, # Graph to plot
				# 	# 				vel = velTestValid, # Velocity arrows to plot
				# 	# 				videoSpeed = 5,
				# 	# 				videoNum= "x_1900",if_gif=False) # Change speed of animation
				# 	#
				#
				# 	if not os.path.exists(log_dir):
				# 		os.makedirs(log_dir)
				# 	videoName = 'sampleTrajectory.gif'
				# 	saveDir = os.path.join(log_dir, videoName)
				#
				#
				# 	data.make_gif(posTestValid[0], saveDir,fps=60)
				# 	logger_ = Logger(log_dir)
				#
				#
				# 	stats_ = data.evaluate_all(pos=posTestValid, \
				# 	vel = velTestValid)
				#
				#
				#
				# 	for i in range(len(stats_)):
				# 		logs = OrderedDict()
				# 		data.perform_logging(i, logs, logger_, stats_[i], False)
