from instruction import Instruction

import random
from uuid import uuid4
from typing import List, Dict
import numpy as np
from parameters import Parameters

from action import Action
from copy import deepcopy

class Program:
	def __init__(self):

		#: A unique id given to the program for identification
		self.id: UUID = str(uuid4())

		#: A set of registers modified during program execution. This is emergent behaviour.
		self.registers: np.array = np.zeros(Parameters.NUM_REGISTERS)

		#: A program is assigned a random action when created.
		self.action: Action = Action(Instruction(isForcedMemoryInstruction=True))

		#: A list of randomly generated instructions varying in size from 1 to MAX_INSTRUCTION_COUNT.
		self.instructions: List[Instruction] = []

		# Generate a list of instructions ranging from 1 to the maximum number of instructions
		for _ in range(random.randint(1, Parameters.MAX_INSTRUCTION_COUNT)):
			self.instructions.append(Instruction())

	def __str__(self) -> str:
		"""
		Generates a human readable string of the program
		consisting of all the instructions.
		:return: A human readable string of the program
		"""
		header: str = f"Program {self.id}:\n"
		instructions: str = '\n'.join(map(str, self.instructions))
		return f"{header}{instructions}"

	def __hash__(self) -> int:
		"""
		Generates a hash for the program. This is used to compare
		whether two programs are the same. If they have the same hash,
		they are the same program. During mutation, hashes are applied
		when a program is mutated to guarantee that the mutated program
		is distinct from the original.
		:return: A unique hash representing the program
		"""
		return hash(str(self))
	
	def execute(self, memory: List[float], state: np.array) -> None:
		"""
		Execute all of the program's instructions sequentially

		:param state: The feature vector representing the state/observation from the environment.
		""" 
		for instruction in self.instructions:
			instruction.execute(memory, state, self.registers)

	def bid(self, memory: List[float], state: np.array) -> Dict[float, str]:
		"""
		Produces a bid which consists of an action and a confidence value of how
		certain the program is that the action is correct.
		
		The bidding behaviour for a program is an emergent property.

		The confidence value is simply the value stored in the program's first register.
		
		:param state: The feature vector representing the state/observation from the environment

		:return: A dictionary containing a confidence value and a suggested action.
		"""
		self.execute(memory, state)
		
		return {
			"confidence": self.registers[0],
			"action": self.action.value
		}

	def copy(self):
		return deepcopy(self)
