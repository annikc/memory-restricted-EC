# write experiment class
# expt class should take agent and environment
# functions for stepping through events/trials, updating,
# collecting data, writing data
# Annik Carson July 2021
# =====================================
#           IMPORT MODULES            #
# =====================================
import numpy as np
import time
import pickle, csv
import uuid
import torch
import os

class expt(object):
	def __init__(self, agent, environment, **kwargs):
		self.env = environment
		self.agent = agent
		# self.rep_learner = rep_learner  #TODO add in later
		self.data = self.reset_data_logs()
		self.agent.counter = 0
		self.pol_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		self.val_grid = np.empty(self.env.shape)

	def update_ledger(self, parent_folder, file_name, info_list):
		with open(parent_folder + file_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(info_list)

	def save_objects(self, parent_folder, save_id):
		# save data
		with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
			pickle.dump(self.data, savedata)
		# save agent weights
		torch.save(self.agent.MFC.state_dict(), f=f'{parent_folder}agents/{save_id}.pt')
		# save episodic dictionary
		if self.agent.EC != None:
			with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
				pickle.dump(self.agent.EC.cache_list, saveec)

	def record_log(self, env_name, representation_type, n_trials, n_steps, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name      = kwargs.get('file', 'test_bootstrap.csv')
		load_from     = kwargs.get('load_from', ' ')
		mock_log      = kwargs.get('mock_log', False)

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		field_names = [
		'timestamp', #datetime experiment recorded
		'save_id',  # uuid
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'env_name',  # str
		'representation', # str
		'MF_input_dims',  # arch
		'MF_fc1_dims',  # bool
		'MF_fc2_dims',  # bool
		'MF_lr',  # list
		'MF_temp',  # list
		'MF_gamma',  # float
		'EC_cache_limit',  # float
		'EC_temp',  # torch optim. class
		'EC_mem_decay',  # # string
		'EC_use_pvals',  # bool
		'EC_similarity_meas', # string
		'extra_info'
		]
		run_data = [timestamp, save_id, load_from, n_trials, n_steps, env_name, representation_type]
		network_keys = ['input_dims', 'fc1_dims', 'fc2_dims', 'lr', 'temperature']
		ec_keys = ['cache_limit', 'mem_temp', 'memory_envelope', 'use_pvals']
		agent_data = [self.agent.MFC.__dict__[k] for k in network_keys] + [self.agent.gamma]
		if self.agent.EC != None:
			ec_data = [self.agent.EC.__dict__[k] for k in ec_keys]
			ec_data.append(self.agent.EC.__dict__['distance_metric'])
		else:
			ec_data = ["None" for k in ec_keys] + ["None"]

		extra_info = kwargs.get('extra', [])

		log_jam = run_data + agent_data + ec_data + extra_info

		# write to logger
		if not os.path.exists(parent_folder+log_name):
			self.update_ledger(parent_folder,log_name,info_list=field_names)

		if mock_log:
			log_jam = log_jam+['mock log']

		self.update_ledger(parent_folder,log_name,info_list=log_jam)

		if not mock_log: ## can turn on flag to write to csv without saving files
			self.save_objects(parent_folder,save_id)
		print(f'Logged with ID {save_id}')

	def reset_data_logs(self):
		data_log = {'total_reward': [],
					'loss': [[], []],
					'trial_length': [],
					'EC_snap': [],
					'P_snap': [],
					'V_snap': []
					}
		return data_log

	def representation_learning(self):
		# TODO
		# to be run before experiment to learn representations of states
		pass

	def snapshot(self, states, observations):
		# initialize empty data frames
		pol_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])
		val_grid = np.empty(self.env.shape)

		mem_grid = np.zeros(self.env.shape, dtype=[(x, 'f8') for x in self.env.action_list])

		# forward pass through network
		pols, vals = self.agent.MFC(observations)

		# populate with data from network
		for s, p, v in zip(states, pols, vals):
			pol_grid[s] = tuple(p.data.numpy())
			val_grid[s] = v.item()

		return pol_grid, val_grid

	def end_of_trial(self, trial, logsnap=False):
		p, v = self.agent.finish_()

		# temp
		if logsnap:
			states = [self.env.oneD2twoD(x) for x in list(self.agent.state_reps.keys())]
			observations = list(self.agent.state_reps.values())
			MF_pols, MF_vals = self.snapshot(states,observations)
			self.data['V_snap'].append(MF_vals)
			self.data['P_snap'].append(MF_pols)
		# /temp

		self.data['total_reward'].append(self.reward_sum) # removed for bootstrap expts
		self.data['loss'][0].append(p)
		self.data['loss'][1].append(v)

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data['total_reward'])
		else:
			self.running_rwdavg = np.mean(self.data['total_reward'][-self.print_freq:])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

	def single_step(self,trial):
		# get representation for given state of env. TODO: should be in agent to get representation?
		state_representation = self.agent.get_state_representation(self.state)
		readable = self.state

		# get action from agent
		action, log_prob, expected_value = self.agent.get_action(state_representation)
		# take step in environment
		next_state, reward, done, info = self.env.step(action)

		# end of event
		target_value = 0
		self.reward_sum += reward

		self.agent.log_event(episode=trial, event=self.agent.counter,
							 state=state_representation, action=action, reward=reward, next_state=next_state,
							 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
							 done=done, readable_state=readable)
		self.agent.counter += 1
		self.state = next_state
		return done

	def run(self, NUM_TRIALS, NUM_EVENTS, **kwargs):
		self.print_freq = kwargs.get('printfreq', 100)
		self.reset_data_logs()
		self.t = time.time()
		logsnap = kwargs.get('snapshot_logging', False)

		for trial in range(NUM_TRIALS):
			self.state = self.env.reset()
			self.reward_sum = 0

			for event in range(NUM_EVENTS):
				done = self.single_step(trial)

				if done:
					break

			self.end_of_trial(trial,logsnap=logsnap)

class conv_expt(expt):
	def __init__(self, agent, environment):
		super().__init__(agent,environment)

	def record_log(self, env_name, representation_type, n_trials, n_steps, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name      = kwargs.get('file', 'test_bootstrap.csv')
		load_from     = kwargs.get('load_from', ' ')
		mock_log      = kwargs.get('mock_log', False)

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		field_names = [
		'timestamp', #datetime experiment recorded
		'save_id',  # uuid
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'env_name',  # str
		'representation', # str
		'MF_input_dims',  # arch
		'MF_hidden_types',
		'MF_hidden_dims',
		'MF_lr',  # list
		'MF_temp',  # list
		'MF_gamma',  # float
		'EC_cache_limit',  # float
		'EC_temp',  # torch optim. class
		'EC_mem_decay',  # # string
		'EC_use_pvals',  # bool
		'EC_similarity_meas', # string
		'extra_info'
		]
		run_data = [timestamp, save_id, load_from, n_trials, n_steps, env_name, representation_type]
		network_keys = ['input_dims', 'hidden_types', 'hidden_dims', 'lr', 'temperature']
		ec_keys = ['cache_limit', 'mem_temp', 'memory_envelope', 'use_pvals']
		agent_data = [self.agent.MFC.__dict__[k] for k in network_keys] + [self.agent.gamma]
		if self.agent.EC != None:
			ec_data = [self.agent.EC.__dict__[k] for k in ec_keys]
			ec_data.append(self.agent.EC.__dict__['distance_metric'])
		else:
			ec_data = ["None" for k in ec_keys] + ["None"]

		extra_info = kwargs.get('extra', [])

		log_jam = run_data + agent_data + ec_data + extra_info

		# write to logger
		if not os.path.exists(parent_folder+log_name):
			with open(parent_folder + log_name, 'a+', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(field_names)

		with open(parent_folder + log_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			if mock_log:
				writer.writerow(log_jam+["mock log"])
			else:
				writer.writerow(log_jam)

		if not mock_log: ## can turn on flag to write to csv without saving files
			# save data
			with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
				pickle.dump(self.data, savedata)
			# save agent weights
			torch.save(self.agent.MFC.state_dict(), f=f'{parent_folder}agents/{save_id}.pt')
			# save episodic dictionary
			if self.agent.EC != None:
				with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
					pickle.dump(self.agent.EC.cache_list, saveec)
		print(f'Logged with ID {save_id}')

class flat_expt(expt):
	def __init__(self, agent, environment):
		super().__init__(agent,environment)

	def record_log(self, env_name, representation_type, n_trials, n_steps, **kwargs): ## TODO -- set up logging
		parent_folder = kwargs.get('dir', './Data/')
		log_name      = kwargs.get('file', 'test_bootstrap.csv')
		load_from     = kwargs.get('load_from', ' ')
		mock_log      = kwargs.get('mock_log', False)

		save_id = uuid.uuid4()
		timestamp = time.asctime(time.localtime())

		field_names = [
		'timestamp', #datetime experiment recorded
		'save_id',  # uuid
		'load_from',  # str
		'num_trials',  # int
		'num_events',  # int
		'env_name',  # str
		'representation', # str
		'MF_input_dims',  # arch
		'MF_lr',  # list
		'MF_temp',  # list
		'MF_gamma',  # float
		'EC_cache_limit',  # float
		'EC_temp',  # torch optim. class
		'EC_mem_decay',  # # string
		'EC_use_pvals',  # bool
		'EC_similarity_meas', # string
		'extra_info'
		]
		run_data = [timestamp, save_id, load_from, n_trials, n_steps, env_name, representation_type]
		network_keys = ['input_dims', 'lr', 'temperature']
		ec_keys = ['cache_limit', 'mem_temp', 'memory_envelope', 'use_pvals']
		agent_data = [self.agent.MFC.__dict__[k] for k in network_keys] + [self.agent.gamma]
		if self.agent.EC != None:
			ec_data = [self.agent.EC.__dict__[k] for k in ec_keys]
			ec_data.append(self.agent.EC.__dict__['distance_metric'])
		else:
			ec_data = ["None" for k in ec_keys] + ["None"]

		extra_info = kwargs.get('extra', [])

		log_jam = run_data + agent_data + ec_data + extra_info

		# write to logger
		if not os.path.exists(parent_folder+log_name):
			with open(parent_folder + log_name, 'a+', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(field_names)

		with open(parent_folder + log_name, 'a+', newline='') as file:
			writer = csv.writer(file)
			if mock_log:
				writer.writerow(log_jam+["mock log"])
			else:
				writer.writerow(log_jam)

		if not mock_log: ## can turn on flag to write to csv without saving files
			# save data
			with open(f'{parent_folder}results/{save_id}_data.p', 'wb') as savedata:
				pickle.dump(self.data, savedata)
			# save agent weights
			torch.save(self.agent.MFC.state_dict(), f=f'{parent_folder}agents/{save_id}.pt')
			# save episodic dictionary
			if self.agent.EC != None:
				with open(f'{parent_folder}ec_dicts/{save_id}_EC.p', 'wb') as saveec:
					pickle.dump(self.agent.EC.cache_list, saveec)
		print(f'Logged with ID {save_id}')

class flat_random_walk(flat_expt):
	def __init__(self, agent, environment):
		super().__init__(agent,environment)

	def single_step(self,trial):
		# get representation for given state of env. TODO: should be in agent to get representation?
		state_representation = self.agent.get_state_representation(self.state)
		readable = self.state

		# get action from agent
		action, log_prob, expected_value = self.agent.get_action(state_representation)
		action = np.random.choice(4)
		# take step in environment
		next_state, reward, done, info = self.env.step(action)

		# end of event
		target_value = 0
		self.reward_sum += reward

		self.agent.log_event(episode=trial, event=self.agent.counter,
							 state=state_representation, action=action, reward=reward, next_state=next_state,
							 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
							 done=done, readable_state=readable)
		self.agent.counter += 1
		self.state = next_state
		return done

class flat_dist_return(flat_expt):
	def __init__(self,agent,envrionment):
		super().__init__(agent,envrionment)
		self.data['dist_rtn'] = []
		self.print_flag = True

	def single_step(self,trial):
		# get representation for given state of env.
		state_representation = self.agent.get_state_representation(self.state)
		readable = self.state

		# get action from agent
		action, log_prob, expected_value, distance, ec_readable = self.agent.get_action(state_representation)
		#get distance from EC

		# take step in environment
		next_state, reward, done, info = self.env.step(action)

		# end of event
		target_value = 0
		self.reward_sum += reward

		self.agent.log_event(episode=trial, event=self.agent.counter,
							 state=state_representation, action=action, reward=reward, next_state=next_state,
							 log_prob=log_prob, expected_value=expected_value, target_value=target_value,
							 done=done, readable_state=readable, distance=distance, ec_readable=ec_readable)
		self.agent.counter += 1
		self.state = next_state
		return done

	def end_of_trial(self, trial, logsnap=False):
		p, v = self.agent.finish_()

		if self.print_flag:
			print(len(self.agent.EC.cache_list.keys()))

		if len(self.agent.EC.cache_list.keys()) == self.agent.EC.cache_limit and self.print_flag:
			print(f"cache limit hit on trial {trial}")
			self.print_flag = False

		# temp
		if logsnap:
			states = [self.env.oneD2twoD(x) for x in list(self.agent.state_reps.keys())]
			observations = list(self.agent.state_reps.values())
			MF_pols, MF_vals = self.snapshot(states,observations)
			self.data['V_snap'].append(MF_vals)
			self.data['P_snap'].append(MF_pols)
		# /temp

		self.data['total_reward'].append(self.reward_sum) # removed for bootstrap expts
		self.data['loss'][0].append(p)
		self.data['loss'][1].append(v)
		self.data['dist_rtn'].append(self.agent.avg_dist_rtn[-1])

		if trial <= 10:
			self.running_rwdavg = np.mean(self.data['total_reward'])
		else:
			self.running_rwdavg = np.mean(self.data['total_reward'][-self.print_freq:])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

class flat_ec_pol_track(flat_expt):
	def __init__(self,agent,envrionment):
		super().__init__(agent,envrionment)
		self.data['ec_dicts'] = []
		self.print_flag = True

	def end_of_trial(self, trial,logsnap):
		p, v = self.agent.finish_()

		self.data['total_reward'].append(self.reward_sum) # removed for bootstrap expts
		self.data['loss'][0].append(p)
		self.data['loss'][1].append(v)

		self.data['ec_dicts'].append(self.agent.EC.cache_list.copy())


		if trial <= 10:
			self.running_rwdavg = np.mean(self.data['total_reward'])
		else:
			self.running_rwdavg = np.mean(self.data['total_reward'][-self.print_freq:])

		if trial % self.print_freq == 0:
			print(f"Episode: {trial}, Score: {self.reward_sum} (Running Avg:{self.running_rwdavg}) [{time.time() - self.t}s]")
			self.t = time.time()

