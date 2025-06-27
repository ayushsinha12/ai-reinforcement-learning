import time
import pickle
import numpy as np
import random
from vis_gym import *


gui_flag = True # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g

'''

Complete the function below to do the following:

	1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
	   configuration and taking actions until a terminal state is reached.
	2. Instead of saving all gameplay history, maintain and update Q-values for each state-action pair that your agent encounters in a dictionary.
	3. Use the Q-values to select actions in an epsilon-greedy manner. Refer to assignment instructions for a refresher on this.
	4. Update the Q-values using the Q-learning update rule. Refer to assignment instructions for a refresher on this.

	Some important notes:
		
		- The state space is defined by the player's position (x,y), the player's health (h), and the guard in the cell (g).
		
		- To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
		  For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
		  will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

		- Your Q-table should be a dictionary with the following format:

				- Each key is a number representing the state (hashed using the provided hash() function), and each value should be an np.array
				  of length equal to the number of actions (initialized to all zeros).

				- This will allow you to look up Q(s,a) as Q_table[state][action], as well as directly use efficient numpy operators
				  when considering all actions from a given state, such as np.argmax(Q_table[state]) within your Bellman equation updates.

				- The autograder also assumes this format, so please ensure you format your code accordingly.
  
		  Please do not change this representation of the Q-table.
		
		- The four actions are: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT), 4 (FIGHT), 5 (HIDE)

		- Don't forget to reset the environment to the initial configuration after each episode by calling:
		  obs, reward, done, info = env.reset()

		- The value of eta is unique for every (s,a) pair, and should be updated as 1/(1 + number of updates to Q_opt(s,a)).

		- The value of epsilon is initialized to 1. You are free to choose the decay rate.
		  No default value is specified for the decay rate, experiment with different values to find what works.

		- To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
		  Example usage below. This function should be called after every action.
		  if gui_flag:
		      refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	Finally, return the dictionary containing the Q-values (called Q_table).

'''

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
	"""
	Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon is decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
	n_states  = env.grid_size * env.grid_size * len(env.health_states) * (len(env.guards) + 1)
	n_actions = env.action_space.n
	num_updates = np.zeros((n_states, n_actions), dtype=int)

	Q_table = {}

	for ep in range(num_episodes):
		#print(ep)
		#if ep in (5000, 7000, 10000, 15000, 25000, 30000, 50000, 60000, 80000, 90000, 95000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000): 
			#print(ep)
        
		obs, reward, done, info = env.reset()
		#print(ep)
        
		while not done:
			s = hash(obs)

			if s not in Q_table:
				Q_table[s] = np.zeros(n_actions)

            # the epsilon-greedy action selection
			if np.random.rand() < epsilon:
				a = env.action_space.sample()
			else:
				a = int(np.argmax(Q_table[s]))

            # takes action
			next_obs, r, done, info = env.step(a)
			s_next = hash(next_obs)

            
			if s_next not in Q_table:
				Q_table[s_next] = np.zeros(n_actions)

            # updating the rule
			# used chatgpt to perfect my calculations for the math 
			eta = 1.0 / (1 + num_updates[s, a])
			target = r + gamma * np.max(Q_table[s_next])
			Q_table[s][a] = (1 - eta) * Q_table[s][a] + eta * target
			num_updates[s, a] += 1

			if gui_flag:
				refresh(next_obs, r, done, info)
			
			obs = next_obs

		epsilon *= decay_rate
		
	return Q_table

decay_rate = 0.9999

Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.

Comment before final submission or autograder may fail.
'''

# Q_table = np.load('Q_table.pickle', allow_pickle=True)

# obs, reward, done, info = env.reset()
# total_reward = 0
# while not done:
# 	state = hash(obs)
# 	action = np.argmax(Q_table[state])
# 	obs, reward, done, info = env.step(action)
# 	total_reward += reward
# 	if gui_flag:
# 		refresh(obs, reward, done, info)  # Update the game screen [GUI only]

# print("Total reward:", total_reward)

# # Close the
# env.close() # Close the environment

'''
Q_table = np.load('Q_table.pickle', allow_pickle=True)
obs, reward, done, info = env.reset()
total_reward = 0
while not done:
    state = hash(obs)
    action = np.argmax(Q_table[state])
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if gui_flag:
        refresh(obs, reward, done, info)
print("Total reward:", total_reward)
env.close()
'''

# test for code generated by chatgpt is below 
'''
if __name__ == '__main__':
	print("Entering main…")
    # 1) Make results reproducible
	random.seed(0)
	np.random.seed(0)

    # 2) Smoke test: small run to populate Q-table
	print("Running smoke test (100 episodes)…")
	Q_small = Q_learning(num_episodes=10, gamma=0.9, epsilon=1, decay_rate=0.99)
	print(f"  → States visited: {len(Q_small)}")
	for s in list(Q_small)[:5]:
		print(f"    state {s}: {Q_small[s]}")

    # 3) Structural assertions
	assert all(isinstance(v, np.ndarray) and v.shape == (env.action_space.n,)
               for v in Q_small.values()), "Each Q[s] must be a NumPy array of length 6"
	max_state = env.grid_size*env.grid_size*len(env.health_states)*(len(env.guards)+1)
	assert all(0 <= s < max_state for s in Q_small.keys()), "Invalid state IDs in Q-table"

    # 4) Policy comparison
	def run_policy(Q_table, runs=50):
		total_return = 0
		for _ in range(runs):
			obs, r, done, _ = env.reset()
			cum = 0
			while not done:
				s = hash(obs)
				if s in Q_table:
					a = int(np.argmax(Q_table[s]))
				else:
					a = env.action_space.sample()
				obs, r, done, _ = env.step(a)
				cum += r
			total_return += cum
		return total_return / runs

	avg_random  = run_policy({}, runs=50)
	avg_learned = run_policy(Q_small, runs=50)
	print(f"Avg random policy return  = {avg_random:.1f}")
	print(f"Avg learned policy return = {avg_learned:.1f}")

    # 5) (Optional) Save the small Q-table for manual inspection
	with open('Q_small.pickle', 'wb') as f:
		pickle.dump(Q_small, f, protocol=pickle.HIGHEST_PROTOCOL)
	print("Smoke test complete. Remove test code and bump num_episodes for full training.")
'''
