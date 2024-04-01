from program import Program
from parameters import Parameters
from memory import Memory
from team import Team
from instruction import Instruction

from typing import List
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
import math
import uuid

class Debugger:
	"""
	The Debugger classes provides functionality to display information
	about the team and program population as well as the ability to visualize
	the policy graphs created by root teams.
	"""

	@staticmethod
	def plotTeam(_team: Team, teamPopulation, ax) -> None:
		"""
		Display the policy graph for each root team.
		"""
		G = nx.DiGraph()

		def process_team(team):
			visited.add(str(team.id))

			# Add team as a node with a different color and black outline
			if team.referenceCount == 0:
				G.add_node(team.id, color='black', node_type='team', root=True, edgecolor='black', linewidth=2)
			else:
				G.add_node(team.id, color='white', node_type='team', root=False, edgecolor='black', linewidth=2)

			for program in team.programs:
				# Add program as a node with custom shape and different color
				if type(program.action.value) is Instruction:
					if program.action.id in team.policy:
						G.add_node(program.id, shape='{}', color="red", node_type='program')
					else:
						G.add_node(program.id, shape='{}', color="lightgray", node_type='program')

					actionId: str = str(uuid.uuid4())
					G.add_edge(team.id, program.id) 
					G.add_node(actionId, color='white', node_type='action', label=str(program.action.value))
					G.add_edge(program.id, actionId)
				else:
					for childTeam in teamPopulation:
						if program.action.value != team.id and childTeam.id == str(program.action.value) and str(childTeam.id) not in visited:
							actionId: str = str(uuid.uuid4())
							G.add_node(actionId, shape='{}', color='lightgray', node_type='program')
							G.add_edge(actionId, childTeam.id)
							G.add_edge(team.id, actionId)
							process_team(childTeam)

		visited = set()

		process_team(_team)

		pos = graphviz_layout(G, prog='neato')

		labels = nx.get_edge_attributes(G, 'label')

		node_colors = [G.nodes[node]['color'] for node in G.nodes]
		edge_colors = [G.nodes[node]['edgecolor'] if G.nodes[node]['node_type'] == 'team' else 'none' for node in G.nodes]
		linewidths = [G.nodes[node]['linewidth'] if G.nodes[node]['node_type'] == 'team' else 0 for node in G.nodes]
		node_shapes = nx.get_node_attributes(G, 'shape')
		node_types = nx.get_node_attributes(G, 'node_type')

		# Set different node sizes for team and program nodes
		node_size = [700 if node_types[node] == 'team' else 350 for node in G.nodes]

		# Use label=None to remove labels for program nodes
		labels_dict = {node: '' if node_types[node] == 'program' or node_types[node] == 'action' else node for node in G.nodes}

		nx.draw(G, pos, ax=ax, with_labels=True, labels=labels_dict, node_size=node_size,
				node_color=node_colors, edgecolors=edge_colors, linewidths=linewidths, font_size=8)

		# Manually draw action nodes with text label
		action_nodes = [node for node in G.nodes if G.nodes[node]['node_type'] == 'action']
		for action_node in action_nodes:
			bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7, lw=1)  # Adjust the boxstyle and other parameters as needed
			ax.text(pos[action_node][0], pos[action_node][1], G.nodes[action_node]['label'], ha='center', va='center', color='black', size=6, bbox=bbox_props)

		# Manually draw program nodes with custom shape
		program_nodes = [node for node in G.nodes if G.nodes[node]['node_type'] == 'program']
		for program_node in program_nodes:
			ax.text(pos[program_node][0], pos[program_node][1], G.nodes[program_node]['shape'], ha='center', va='center', color='black', size=8)

	@staticmethod
	def plotMemory(team: Team, ax) -> None:
		# Reshape the list into a square matrix
		size = int(len(team.memory) ** 0.5)
		matrix = np.array(team.memory).reshape((size, size))

		# Round the floats to two decimal places
		matrix_rounded = np.round(matrix, decimals=5)

		# Plot the matrix
		ax.matshow(matrix_rounded, cmap='viridis')

		# Add text annotations
		for i in range(size):
			for j in range(size):
				ax.text(j, i, str(matrix_rounded[i, j]), va='center', ha='center')
