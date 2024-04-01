from program import Program, Instruction
from team import Team
import random
from typing import List
from parameters import Parameters
from action import Action

class Mutator:

	@staticmethod
	def mutateInstruction(instruction: Instruction) -> None:
		"""
		An instruction can be mutated by:

		- Changing its input source (program registers or state/observations from the environment).

		- Changing its operation

		- Changing the source register

		- Changing the destination register

		It is guaranteed that a mutation will produce an instruction distinct from the original.

		:param instruction: the instruction to mutate
		"""
		parts: List[str] = [ "ADDRESSING_MODE", "OPERATION", "SOURCE REGISTER", "DESTINATION REGISTER"]

		mutatedPart: str = random.choice(parts)

		originalHash: int = hash(instruction)
		
		if mutatedPart == "ADDRESSING_MODE":
			instruction.mode = random.choice(["INPUT", "REGISTERS", "MEMORY"])

			# ensure that the registers remain within bounds
			if instruction.mode == "INPUT" and instruction.source > Parameters.NUM_OBSERVATIONS - 1:
				instruction.source = instruction.source % Parameters.NUM_OBSERVATIONS
			elif instruction.mode == "REGISTERS" and instruction.source > Parameters.NUM_REGISTERS - 1:
				instruction.source = instruction.source % Parameters.NUM_REGISTERS
			elif instruction.mode == "MEMORY" and instruction.source > Parameters.MEMORY_SIZE - 1:
				instruction.source = instruction.source % Parameters.MEMORY_SIZE
				
		elif mutatedPart == "OPERATION":
			instruction.operation = random.choice(['+', '-', '*', "=", '/', 'COS', 'NEGATE', 'NONE'])

		elif mutatedPart == "SOURCE REGISTER":
			if instruction.mode == "INPUT":
				instruction.source = random.randint(0, Parameters.NUM_OBSERVATIONS - 1)
			elif instruction.mode == "REGISTERS":
				instruction.source = random.randint(0, Parameters.NUM_REGISTERS - 1)
			elif instruction.mode == "MEMORY":
				instruction.source = random.randint(0, Parameters.MEMORY_SIZE - 1)

		elif mutatedPart == "DESTINATION REGISTER":
			if instruction.updateMemory:
				instruction.destination = random.randint(0, Parameters.MEMORY_SIZE - 1) 
			else:
				instruction.destination = random.randint(0, Parameters.NUM_REGISTERS - 1) 
			
		newHash: int = hash(instruction)
		
		# Mutation is the same as the original, try again.
		if newHash == originalHash:
			Mutator.mutateInstruction(instruction)

	@staticmethod
	def mutateProgram(program: Program) -> None:
		"""
		A program can be mutated by:

		- Deleting an instruction
		
		- Adding an instruction
		
		- Swapping an instruction
		
		- Mutating an instruction

		An instruction will be deleted iff there is more than one instruction in the program.

		An instruction will be added iff the instruction count would not exceed the maximum.

		It is guaranteed that the mutated program will be distinct from original.

		:param program: the program to mutate.
		"""
		originalHash: int = hash(program)

		# delete an instruction
		if len(program.instructions) > 1:
			if random.random() < Parameters.DELETE_INSTRUCTION_PROBABILITY:
				program.instructions.remove(random.choice(program.instructions))

		# add an instruction
		if random.random() < Parameters.ADD_INSTRUCTION_PROBABILITY:
			if len(program.instructions) < Parameters.MAX_INSTRUCTION_COUNT:
				program.instructions.append(Instruction())

		# swap an instruction
		if random.random() < Parameters.SWAP_INSTRUCTION_PROBABILITY:
			if len(program.instructions) > 2:
				i, j = random.sample(range(len(program.instructions)), 2)
				program.instructions[i], program.instructions[j] = program.instructions[j], program.instructions[i]

		# mutate an instruction
		if random.random() < Parameters.MUTATE_INSTRUCTION_PROBABILITY:
			Mutator.mutateInstruction(random.choice(program.instructions))

		newHash: int = hash(program)

		# Mutation is the same as the original, try again.
		if newHash == originalHash:
			Mutator.mutateProgram(program)

	# TODO: Add a hash for teams so we know each team is unique after mutation
	@staticmethod
	def mutateTeam(programPopulation: List[Program], teamPopulation: List[Team], team: Team):
		"""
		A team can be mutated by:

		- Adding an existing program

		- Deleting a program

		- Creating a new program and assigning it to the team

		- Changing a program's action to a new action

		- Changing a program's action to reference another team

		An existing program will be added iff the program isn't already used by the team.
		
		A program will be deleted iff the program has at least one program remaining.

		It is guaranteed that after mutation the team will have at least one atomic action.
		
		:param programPopulation: all programs at time of mutation
		:param teamPopulation: all teams at time of mutation
		:param team: the team to mutate
		"""
		
		# add a program
		if random.random() < Parameters.ADD_PROGRAM_PROBABILITY:
			if len(team.programs) < Parameters.MAX_PROGRAM_COUNT:
				newProgram: Program = random.choice(programPopulation) 

				ids = [ program.id for program in team.programs ]
				# Ensure we're not adding a duplicate program to the team
				if len(programPopulation) > len(ids):
						while newProgram.id in ids:
							newProgram = random.choice(programPopulation)
							team.programs.append(newProgram)

		# delete a program
		numAtomicActions: int = 0
		for program in team.programs:
			if type(program.action.value) is Instruction:
				numAtomicActions += 1

		if random.random() < Parameters.DELETE_PROGRAM_PROBABILITY:
			if numAtomicActions > 2:
				team.programs.remove(random.choice(team.programs))
				
		# create a new program
		if random.random() < Parameters.NEW_PROGRAM_PROBABILITY:
			if len(team.programs) < Parameters.MAX_PROGRAM_COUNT:
				program: Program = Program()
				programPopulation.append(program)
				team.programs.append(program)

		# mutate the team's programs
		for program in team.programs:
			Mutator.mutateProgram(program)
			
		# mutate a program's action
		if random.random() < Parameters.MUTATE_PROGRAM_PROBABILITY:
			
			program: Program = random.choice(team.programs)

			numAtomicActions: int = 0
			for program in team.programs:
				if type(program.action.value) is Instruction:
					numAtomicActions += 1

			# A team must have at least one atomic action!
			if numAtomicActions > 1 and random.random() < Parameters.TEAM_POINTER_PROBABILITY:
				newTeam: Team = random.choice(teamPopulation)
				while newTeam.id == team.id:
					newTeam = random.choice(teamPopulation)

				newTeam.referenceCount += 1    
				program.action = Action(str(newTeam.id))
			else:
				#if program.action.value not in Parameters.ACTIONS:
				#	for t in teamPopulation:
				#		if str(t.id) == program.action.value:
							#t.referenceCount -= 1
							
				program.action = Action(Instruction(isForcedMemoryInstruction=True))
