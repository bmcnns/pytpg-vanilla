from model import Model
import random
from typing import List
from environment import Environment 
from debugger import Debugger
import sys
from instruction import Instruction
from program import Program
from parameters import Parameters
from team import Team

import argparse

import matplotlib.pyplot as plt

def main():
	# Determine TPG's configuration from cmdline arguments
	parser = argparse.ArgumentParser("Canonical TPG")
	parser.add_argument("--environment", type=str, help="The environment to benchmark against (e.g. Frostbite, LunarLander-v2)", required=True)
	parser.add_argument("--population_size", type=int, help="The number of root teams in every generation", required=True)
	parser.add_argument("--actions", type=str, help="A comma separated list of actions used in the environment. Omit this argument for Atari", required=False)
	parser.add_argument("--num_observations", type=int, help="The size of the input/feature space", required=True)
	parser.add_argument("--num_generations", type=int, help="The number of generations the model will be trained for.", required=True)
	parser.add_argument("--max_steps", type=int, help="The number of interactions an agent can have with its environment before termination", required=True)
	parser.add_argument("--generation", type=int, help="The number that the generation counter should be initialized to (helpful when loading/saving a model", default=1,required=False)
	parser.add_argument("--model", type=str, help="The path to the model file to load.", required=False)
	args = parser.parse_args()

	# Update the parameters with the ones we read from the CLI
	Parameters.ENVIRONMENT = args.environment
	Parameters.POPULATION_SIZE = args.population_size
	
	if args.actions is not None:
		Parameters.ACTIONS = args.actions.replace(" ", "").split(',')
	else:
		Parameters.ACTIONS = range(18)

	Parameters.NUM_OBSERVATIONS = args.num_observations

	# Initialize the OpenAI Gym / Atari Learning Environment API
	environment = Environment(Parameters.ENVIRONMENT, Parameters.ACTIONS)
	
	# If specified, load a previously trained model
	if args.model == None:
		model = Model()

	else:
		model = Model.load(f"{args.model}")

	# Train the model
	model.fit(environment=environment, numGenerations=args.num_generations, maxStepsPerGeneration=args.max_steps, startingGeneration=args.generation)

if __name__ == "__main__":
	main()
