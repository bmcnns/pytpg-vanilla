import numpy as np
import copy

from parameters import Parameters

class QLearner:

	def __init__(self, numStates, numActions, gamma, learningRate):
		rng = np.random.default_rng()
		self.numStates = numStates
		self.numActions = numActions
		self.weights = rng.normal(0, 1, size=(self.numStates+1, self.numActions))
		self.gamma = gamma
		self.learningRate = learningRate

	def train(self, previousState, nextState, reward, action):
		previousInput = np.zeros(self.numStates + 1)
		previousInput[0] = -1
		previousInput[1:] = previousState

		nextInput = np.zeros(self.numStates + 1)
		nextInput[0] = -1
		nextInput[1:] = nextState

		# Calculate the current Q-value
		currentQ = np.dot(self.weights.T, previousInput)

		# Calculate the target Q-value using Q-learning
		maxNextQ = np.max(np.dot(self.weights.T, nextInput))
		targetQ = currentQ.copy()
		targetQ[action] = reward + self.gamma * maxNextQ

		error = targetQ - currentQ
		self.weights += self.learningRate * np.outer(previousInput, error)

	def predict(self, state):
		input = np.zeros(self.numStates + 1)
		input[0] = -1
		input[1:] = state
		return np.dot(self.weights.T, input)
		
	def copy(self):
		copied_instance = QLearner(self.numStates, self.numActions, self.gamma, self.learningRate)
		copied_instance.weights = copy.deepcopy(self.weights)
		return copied_instance
