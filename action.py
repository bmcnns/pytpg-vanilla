from uuid import uuid4
from memory_instruction import MemoryInstruction

class Action:
	def __init__(self, value):
		# This is the memory specific implementation --
		# therefore, instead of having an atomic action
		# now actions are memory instructions

		# A value may either be a memory instruction
		# or a reference to another team by id.
		self.id: str = str(uuid4()) 
		self.value = value

	def __str__(self) -> str:
		return str(self.value)
