from typing import Tuple, List
import numpy as np
from ale_py import ALEInterface
from ale_py.roms import Frostbite, UpNDown
from atari import Atari
import gym
import random

import matplotlib.pyplot as plt

class Environment:
	"""
	This class is a wrapper and an interface to provide common access to a reset method and a step method. This is used to abstract away the differences between environments.
	"""
	def __init__(self, environment: str, actions: List):
		if environment in [ "Frostbite", "UpNDown" ]:
			self.environment = ALEEnvironment(environment, range(18))
		else:
			self.environment = GymEnvironment(environment, actions) 

	def reset(self) -> np.array:
		return self.environment.reset()

	def step(self, action: str) -> Tuple[np.array, float, bool]:
		return self.environment.step(action)

	def display(self, ax, generation, team, step, score):
		return self.environment.display(ax, generation, team, step, score)

class GymEnvironment():
	"""
	This class implements the Environment interface for the OpenAI Gymnasium.
	"""
	def __init__(self, environment: str, actions: List):
		if environment == "CarRacing-v2":
			self.gym = gym.make(environment, render_mode='rgb_array', domain_randomize=False, continuous=False)
		else:
			self.gym = gym.make(environment, render_mode='rgb_array')

		self.environment = environment
		self.actions = actions
		
	def reset(self) -> np.array:
		"""
		Reset the environment back to its initial state.

		:return: the initial state/observation.
		"""
		return np.array(self.gym.reset()[0]).flatten()

	def step(self, action: str) -> Tuple[np.array, float, bool]:
		"""
		Do the provided action in the environment

		:param action: the action to take
		:return: the state/observation after the action, the reward associated with the action, whether or not the agent has reached a terminal state
		"""
		index: int = self.actions.index(action)
		state, reward, isTerminated, isTruncated, _ = self.gym.step(index)
		return (np.array(state).flatten(), reward, isTerminated or isTruncated)

	def display(self, ax, generation, team, step, score):
		ax.imshow(self.gym.render())
		ax.set_title(f"{self.environment} Generation #{generation} Step {step} Score {score}")
		plt.axis('off')


class ALEEnvironment():
	"""
	This class implements the Environment interface for the Atari Learning Environment (ALE).
	"""
	def __init__(self, environment: str, actions: List[str]):
		self.ale = ALEInterface()
		
		if environment == "Frostbite":
			return self.ale.loadROM(Frostbite)
		elif environment == "UpNDown":
			return self.ale.loadROM(UpNDown)
		else:
			raise RuntimeError("Environment could not be loaded: ROM not found.")

	def reset(self) -> np.array:
		"""
		Reset the environment back to its initial state.

		:return: the initial state/observation.
		"""
		self.ale.reset_game()
		return Atari.preprocess(self.ale.getScreenRGB())

	def step(self, action: int) -> Tuple[np.array, float, bool]:
		"""
		Do the provided action in the environment

		:param action: the action to take
		:return: the state/observation after the action, the reward associated with the action, whether or not the agent has reached a terminal state
		"""
		reward = self.ale.act(action)
		return (Atari.preprocess(self.ale.getScreenRGB()), reward, self.ale.game_over())

	def display(self, ax, generation, team, step, score):
		ax.imshow(self.ale.getScreenRGB())
		ax.set_title(f"{self.environment} Generation #{generation} Step {step} Score {score}")
		plt.axis('off')

	
