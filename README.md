# Episodic Control with Restricted Memory Capacity
Code used to generate and analyze data for Episodic Control in RL problems with restricted memory capacity to simulate forgetting. 

## Simulations
Code to run simulations for different forgetting conditions. Simulation files use elements specified in modules folder, including environments, agents, and episodic control modules, as well as experiments set up within modules/Experiments. Simulation files only contain code to gather these elements for a simulation run. Results of simulation runs are saved to Data folder. On git repo no data is saved, but can be provided upon request. 

## Data
After simulations are run, the function record_log saves different elements of the run separately. (1) The dictionary of results which includes total reward achieved on each episode, as well as other elements specific to each experiment. (2) The episodic controller cache list, saved as a dictionary, which contains the information that episodic controller had in memory at the last recorded episode, and not prior history. (3) The weights of the model-free actor critic agent, so that this agent can be reconstructed later. For this paper, the details of a model-free controller were not relevant, so only a dummy network was provided to the agent class. 

## Analysis
Reads saved files from Data folder and generates plots of results. Most plotting is done with functions outlined in Analysis/analysis_utils.py


