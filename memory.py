from typing import List
from parameters import Parameters
import numpy as np

class Memory:
	registers: List[float]

	def __init__(self):
		self.registers = np.random.normal(0, 1, Parameters.MEMORY_SIZE)
