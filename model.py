from environment import Environment
from program import Program
from team import Team
from mutator import Mutator
from parameters import Parameters
from statistics import Statistics
from copy import deepcopy
from memory import Memory
from qlearner import QLearner

#from diversity import Diversity


import pickle
import os

import matplotlib.pyplot as plt

import random
from typing import List, Tuple, Dict
import numpy as np
from uuid import uuid4
from debugger import Debugger

from collections import deque

class Model:
	""" The Model class wraps all Tangled Program Graph functionality into an easy-to-use class. """
	def __init__(self):

		self.id = str(uuid4())

		
		self.statistics = Statistics()
		
		#: The pool of available (competitive) programs 
		self.programPopulation: List[Program] = [ Program() for _ in range(Parameters.INITIAL_PROGRAM_POPULATION)]

		#: The pool of competitive teams 
		self.teamPopulation: List[Team] = [ Team(self.programPopulation) for _ in range(Parameters.POPULATION_SIZE)]

		for team in self.teamPopulation:
			team.referenceCount = 0

	def getRootTeams(self) -> List[Team]:
		"""
		Gets a list of teams that are not referenced by any other team (root teams).

		:return: A list of root teams.
		"""
		rootTeams: List[Team] = []
		for team in self.teamPopulation:
			if team.referenceCount == 0:
				rootTeams.append(team)

		return rootTeams

	def fit(self, environment: Environment, numGenerations: int, maxStepsPerGeneration: int, startingGeneration:int = 1) -> None:
		"""
		Trains the model against a given environment to learn the optimal policy to maximize cumulative reward.

		:param environment: The environment to train against
		:param numGenerations: The number of generations to train the model for.
		:param maxStepsPerGeneration: How many actions the agent can make before time-out.
		"""

		for generation in range(startingGeneration, startingGeneration + numGenerations + 1):
			for teamNum, team in enumerate(self.getRootTeams()):
				#team.qlearner = QLearner(Parameters.MEMORY_SIZE,
				#						 len(Parameters.ACTIONS),
				#						 Parameters.QLEARNER_GAMMA,
				#						 Parameters.QLEARNER_LEARNING_RATE)

				episodicRewards = np.zeros(Parameters.EPISODES)
				episodeStep = 0

				#team.memory = np.random.normal(0, 1, Parameters.MEMORY_SIZE)
								
				for episode in range(Parameters.EPISODES):
					state = environment.reset()

					
					score = 0
					step = 0

					while True:
						registers, instruction = team.getAction(self.teamPopulation, state, visited=[])

						# get the memory before being modified by the genetic algorithm
						previousMemoryState = deepcopy(team.memory)

						# use the q-learner to choose an action
						action = team.qlearner.predict(previousMemoryState).argmax()

						if not Parameters.IS_EVALUATION and random.random() < Parameters.QLEARNER_EPSILON:
							action = random.randint(0, len(Parameters.ACTIONS) - 1)

						# update the memory using the genetic algorithm
						instruction.execute(team.memory, state, registers)

						team.qlearner.predict(team.memory)

						# get the reward and new state after using the q-learner's chosen action
						state, reward, finished = environment.step(Parameters.ACTIONS[action])

						# get the memory after being modified by the genetic algorithm
						newMemoryState = deepcopy(team.memory)

						# update the q-values
						team.qlearner.train(previousMemoryState, newMemoryState, reward, action)

						score += reward
						step += 1
						episodeStep += 1
						
						# If the team did well previously, record a video of its performance

						if Parameters.SAVE_VIDEO:
							if generation % Parameters.SAVE_VIDEO_AFTER_X_GENERATIONS == 0:
								if team.luckyBreaks > 0:
									self.statistics.save(environment, generation, team, self.teamPopulation, episodeStep, score)

						if finished or step == maxStepsPerGeneration:
							break

					episodicRewards[episode] = score
					#print(f"Episode completed with score: {score}")
					
				team.scores.append(np.median(episodicRewards))
				print(f"Generation #{generation} Team #{teamNum + 1} ({team.id})")
				print(f"Team finished with score: {np.median(episodicRewards)}, fitness: {team.getFitness()}")

			print("\nGeneration complete.\n")

			print("Best performing teams:")
			sortedTeams: List[Team] = list(sorted(self.getRootTeams(), key=lambda team: team.getFitness()))

			for team in sortedTeams[-3:]:
				team.luckyBreaks += 1
			for team in sortedTeams[-3:]:
				print(f"Team {team.id} score: {team.getFitness()}, lucky breaks: {team.luckyBreaks}")
				print()

			self.save(f"models/{self.id}/{generation}.pkl")

			self.select()
			self.evolve(generation)


	def cleanProgramPopulation(self) -> None:
		"""
		Used internally. After teams are removed from the population, clean up any programs
		that are no longer in use, since they are no longer competitive.
		"""
		inUseProgramIds: List[str] = []
		for team in self.teamPopulation:
			for program in team.programs:
				inUseProgramIds.append(program.id)

		for program in self.programPopulation:
			if program.id not in inUseProgramIds:
				self.programPopulation.remove(program)

	def select(self) -> None:
		"""
		After agents (root teams) are evaluated in a generation, this method is called
		to sort them by fitness and remove POPGAP percentage of the total root team population.
		The program population is cleaned after teams are removed.
		"""
		sortedTeams: List[Team] = list(sorted(self.getRootTeams(), key=lambda team: team.getFitness()))

		remainingTeamsCount: int = int(Parameters.POPGAP * len(self.getRootTeams()))

		for team in sortedTeams[:remainingTeamsCount]:
			
			if team.luckyBreaks > 0:
				team.luckyBreaks -= 1
				print(f"Tried to remove team {team.id} but they had a lucky break! {team.getFitness()} (remaining breaks: {team.luckyBreaks})")
			else:
				if team.referenceCount == 0:
					print(f"Removing team {team.id} with fitness {team.getFitness()}")
					self.teamPopulation.remove(team)

		self.cleanProgramPopulation() 

	def evolve(self, generation: int) -> None:
		"""
		After removing the uncompetitive teams, clone the remaining competitive root teams
		and apply mutations to the clones until the discarded population is replaced.
		"""
		while len(self.getRootTeams()) < Parameters.POPULATION_SIZE:
			team = random.choice(self.getRootTeams()).copy()

			Mutator.mutateTeam(self.programPopulation, self.teamPopulation, team)
			
			#profile = Diversity.getProfile(self.teamPopulation, team)
			#while profile in Diversity.profiles:
			#	Mutator.mutateTeam(self.programPopulation, self.teamPopulation, team)
			#	profile = Diversity.getProfile(self.teamPopulation, team)
				
			#Diversity.profiles.append(profile)

			self.teamPopulation.append(team)

	def __str__(self) -> str:
		"""
		- Displays all programs and their corresponding actions

		- Displays all teams and the programs they reference
		
		:param model: the model to get information about.

		:return: human-readable information about the program and team populations. 
		"""
		output = f"MODEL {self.id}\n"
		for team in self.teamPopulation:
			if team.referenceCount == 0:
				output += "ROOT TEAM "
			else:
				output += "TEAM "
			output += f"{team.id}\n"

			for program in team.programs:
				output += f"\t{program.id}: {program.action.value}\n"

		return output
		
	def save(self, filename: str) -> None:
		"""
		Saves a model by serializing with Pickle
		Individual teams can't be saved because teams reference other teams.
		"""
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, "wb+") as f:
			pickle.dump(self, f)

	@staticmethod
	def load(filename) -> "Model":
		"""
   		Loads a model from a serialized Pickle file
		:param filename: the file path to the model being loaded.
		"""
		
		with open(filename, "rb") as f:
			return pickle.load(f)
		
