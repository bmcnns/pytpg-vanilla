from typing import List
import random
from program import Program
import numpy as np
from uuid import uuid4
from typing import List
from copy import deepcopy
from parameters import Parameters
from action import Action
from instruction import Instruction
from qlearner import QLearner

import numpy as np

class Team:
	def __init__(self, programPopulation: List[Program]):
		"""
		Generates a team with a number of randomly assigned programs.
		A team has at least two programs but may have up to MAX_INITIAL_TEAM_SIZE programs.
		It is guaranteed that a team has at least two atomic actions.
		:param programPopulation: All programs (at the time of the team's creation)
		:return: A new team
		"""

		#: The unique identifier for the team
		self.id: str = str(uuid4())

		self.memory: List[float] = np.random.normal(0, 1, Parameters.MEMORY_SIZE)#[0.0] * Parameters.MEMORY_SIZE

		self.qlearner = QLearner(numStates=Parameters.MEMORY_SIZE,
								 numActions=len(Parameters.ACTIONS),
								 gamma=Parameters.QLEARNER_GAMMA,
								 learningRate=Parameters.QLEARNER_LEARNING_RATE)

		#: The cumulative rewards assigned to the team for each generation
		self.scores: List[float] = []

		#: The programs assigned to the team
		self.programs: List[Program] = [] 

		actions: List[Action] = []

		self.policy = []

		self.rootTeam = None
		
		self.referenceCount: int = 0

		# reset every time getAction() is called
		self.luckyBreaks: int = 0
		"""
		This is a method of keeping elites. If a team is top-performing in a round, it is awarded a lucky break.
		In a future round, if has uncompetitive fitness and should be removed.
		The lucky break prevents its deletion during that generation.
		A lucky break is removed after use. 
		"""

		size: int = random.randint(2, Parameters.MAX_INITIAL_TEAM_SIZE)

		while len(list(set([action.value for action in actions ]))) < 2:
			self.programs: List[Program] = random.sample(programPopulation, k=size)
			actions: List[Action] = [ program.action for program in self.programs] 

	# Choose the program with the highest confidence
	def getAction(self, teamPopulation: List['Team'], state: np.array, visited: List[str]) -> str:
		"""
		Returns the team's suggested action by choosing the action of the highest-bidding program.
		If the highest-bidding program is a reference to another team, recurse until the action is atomic.
		If the policy graph has a cycle, any team is only visited once.

		:param teamPopulation: All teams
		:param state: The feature vector representing the state/observation
		:param visited: Used internally in recursive calls, ensures this method does not recurse forever when a cycle is encountered.

		:return: an atomic action
		"""
		visited.append(self.id)
		
		sortedPrograms = sorted(self.programs, key=lambda program: program.bid(self.memory, state)['confidence'])

		self.policy = []
		
		for program in sortedPrograms:
			if type(program.action.value) is Instruction:
				if program.action == None:
					raise RuntimeError("A NONE ACTION WAS ENCOUNTERED HERE")

				self.policy.append(program.action.id)
				return [ program.registers, program.action.value ]#.action.value
			else:
				for team in teamPopulation:
					if str(team.id) == program.action.value:
						if str(team.id) not in visited:
							return team.getAction(teamPopulation, state, visited)
						else:
							raise RuntimeException("CYCLE")
							
				raise RuntimeError(f"Team {self.id, type(self.id)} points to team {program.action.value, type(program.action.value)}, and that team does not exist within the population.")

	def getFitness(self):
		"""
		This method is used to define the fitness of a team.
		By default, fitness is the score from the previous generation.

		:return: The fitness score
		"""
		return self.scores[-1]

	def __str__(self) -> str:
		output = f"TEAM {self.id}\n"
		for program in self.programs:
			output += f"{program.id}->{program.action.value}\n"
		return output
	
	# Given a parent team, a new offspring team is cloned and mutated
	def copy(self) -> "Team":
		"""
		Clones an existing team
		
		If the original team has 'lucky breaks', they are not carried over to the cloned team.
		The clone is given a new ID such that no two teams have the same ID.
		
		:return: A new team with identical behaviour to the team that was cloned.
		"""
		clone: 'Team' = deepcopy(self)
		clone.qlearner = self.qlearner.copy()
		clone.memory = deepcopy(self.memory)
		clone.programs = []

		for program in self.programs:
			clone.programs.append(program.copy())
		
		clone.referenceCount = 0
		clone.luckyBreaks = 0
		clone.id = str(uuid4())
		return clone
