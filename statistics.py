from team import Team
from environment import Environment
from debugger import Debugger

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import os


class Statistics:
	
	reward_df = pd.DataFrame()
	reward_df.columns.name = 'Generation'
	reward_df.index.name = 'Team'

	def recordPerformance(self, team: Team, generation: int, score: float):
		self.reward_df.loc[team.id, generation] = score

	def getRewardsVersusTime(self, ax):
		data = self.reward_df.fillna(0)
		data = data.T
		colors = ['black'] * len(data.columns)  # Setting line color to black for all lines

		for col in data.columns:
			ax.plot(data.index, data[col], linestyle=':', marker='x', color='black')  # Using 'x' marker and black color
			# Displaying labels for top 3 performing teams
		max_values = data.iloc[-1].nlargest(3)
		top_teams = max_values.index
		
		for team in top_teams:
			max_val = max_values[team] 
			max_index = data.index[-1]
			ax.text(max_index, max_val, team, ha='center', va='bottom', color='black', fontsize=10)

		ax.set_xlabel('Generation')
		ax.set_ylabel('Cumulative Reward')

		ax.set_title('Cumulative rewards of teams over time')

	def save(self, environment: Environment, generation, team, teamPopulation, step, reward):
		fig, axs = plt.subplots(3, 1, figsize=(10,8))
		axs = axs.flatten()
		environment.display(axs[0], generation, team, step, reward)
		Debugger.plotTeam(team, teamPopulation, axs[1])
		Debugger.plotMemory(team, axs[2])

		filename = f"bin/recordings/{team.id}/{generation}/{step}.png"
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		plt.savefig(filename)
		plt.close(fig)
