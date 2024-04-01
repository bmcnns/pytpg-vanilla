import random
import numpy as np
import math
from parameters import Parameters
from memory import Memory
from typing import List

class Instruction:
	def __init__(self, isForcedMemoryInstruction=False): 
		"""
		An instruction defines an operation on a program's registers.
		An instruction may be an addition, subtraction, multiplication, division, cosine, or negation.
		
		:param forceMemoryInstruction: Should this Instruction be forced to modify external memory (used for actions)
		:return: A new instruction
		"""

		#: Defines whether the instruction should read from (program registers, external state memory, or state/observation space).
		self.mode: str = random.choice(["INPUT", "REGISTERS", "MEMORY"])

		#: Defines what operation the instruction should execute.
		self.operation: str = random.choice(['+', '-', '*', '/', '=', 'COS', 'NEGATE', 'NONE' ])
		
		#: The register that the instruction takes as input.
		self.source: int
		
		if self.mode == "INPUT":
			self.source = random.randint(0, Parameters.NUM_OBSERVATIONS - 1)
		elif self.mode == "REGISTERS":
			self.source: int = random.randint(0, Parameters.NUM_REGISTERS - 1)
		elif self.mode == "MEMORY":
			self.source: int = random.randint(0, Parameters.MEMORY_SIZE - 1)

		#: The register that the instruction should be applied to.
		self.destination: int

		if isForcedMemoryInstruction:
			self.destination = random.randint(0, Parameters.MEMORY_SIZE - 1)
		else:
			self.destination = random.randint(0, Parameters.NUM_REGISTERS - 1)

		# Set a flag to update memory instead of the program's internal registers
		self.updateMemory: bool
		if isForcedMemoryInstruction:
			self.updateMemory = True
		else:
			self.updateMemory = False

	def __str__(self) -> str:
		"""
		When the instruction is cast to a string it will return a human-readable format.

		:return: the instruction casted to a string.
		"""
		address: str
		if self.mode == "INPUT":
			address = "STATE"
		elif self.mode == "REGISTERS":
			address = "R"
		elif self.mode == "MEMORY":
			address = "MEM"

		if self.updateMemory:
			prefix = "MEM"
		else:
			prefix = "R"
		
		if self.operation == "COS":
			return f"{prefix}[{self.destination}] = COS({address}[{self.source}])"
		elif self.operation == "=":
			return f"{prefix}[{self.destination}] = {address}[{self.source}]"
		elif self.operation == "NEGATE":
			return f"IF ({prefix}[{self.destination}] < {address}[{self.source}]) THEN {prefix}[{self.destination}] = -{prefix}[{self.destination}]"
		elif self.operation == "NONE":
			return "NO ACTION"
		else:
			return f"{prefix}[{self.destination}] = {prefix}[{self.destination}] {self.operation} {address}[{self.source}]"


	def __hash__(self) -> int:
		"""
		If the hashes of two instructions match, then they are the same.
		
		This is used during mutation to ensure the mutated instruction
		is unique from the original instruction.

		:return: the instruction's hash
		"""
		return hash((self.updateMemory, self.mode, self.operation, self.source, self.destination))

	def execute(self, memory: List[float], state: np.array, registers: np.array) -> None:
		"""
		Updates a program's registers after executing the instruction.
		If the instruction is a division by zero, the register is set to 0.
		If the instruction causes an overflow/underflow, the register is set to inf/-inf.

		:param state: the feature vector representing the state/observation
		:param registers: the registers belonging to the program executing the instruction
		"""
		if self.mode == "INPUT":
			input = state
		elif self.mode == "REGISTERS":
			input = registers
		elif self.mode == "MEMORY":
			input = memory
			
		if self.updateMemory:
			updatedRegisters = memory 
		else:
			updatedRegisters = registers
			
		if self.operation == '+': 
			updatedRegisters[self.destination] = updatedRegisters[self.destination] + input[self.source]
		elif self.operation == '=':
			updatedRegisters[self.destination] = input[self.source]
		elif self.operation == "-":
			updatedRegisters[self.destination] = updatedRegisters[self.destination] - input[self.source]
		elif self.operation == "*":
			updatedRegisters[self.destination] = updatedRegisters[self.destination] * input[self.source]
		elif self.operation == "/":
			if input[self.source] != 0:
				updatedRegisters[self.destination] = updatedRegisters[self.destination] / input[self.source]
			else:
				updatedRegisters[self.destination] = 0
		elif self.operation == "COS":
			updatedRegisters[self.destination] = math.cos(input[self.source])
		elif self.operation == "NEGATE":
			if updatedRegisters[self.destination] < input[self.source]:
				updatedRegisters[self.destination] = -updatedRegisters[self.destination]
		elif self.operation == "NONE":
			pass

		if updatedRegisters[self.destination] == np.inf:
			updatedRegisters[self.destination] = 0#np.finfo(np.float64).min
		elif updatedRegisters[self.destination] == np.NINF:
			updatedRegisters[self.destination] = 0#np.finfo(np.float64).min
