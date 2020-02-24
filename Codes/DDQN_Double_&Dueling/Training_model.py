"""
@project = DQN
@file = Training_model
@author = 10374
@create_time = 2019/12/04 15:50
"""

from Maze_env import Maze
from RL_brain_Double import D_DQN
from RL_brain_Dueling import DuelingDQN


# traing process with env
def run_maze():
	step = 0
	for episode in range(300):
		# initial observation
		step_counter = 0; sum_r = 0
		observation = env.reset()

		while True:

			# fresh env
			env.render()

			# RL choose action based on observation
			action = RL.choose_action(observation)

			# RL take action and get next observation and reward
			observation_, reward, done = env.step(action)
			sum_r += reward

			# 存储记忆的关键一步
			RL.store_memory(observation, action, reward, observation_)

			if (step > 200) and (step % 5 == 0):
				RL.update_copy_params()

			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				break
			step_counter += 1
			step += 1

		interaction = 'Episode %s: total_steps = %s: total_reward = %s' % (episode + 1, step_counter, sum_r)
		RL.reward_his.append(sum_r)
		print('\r{}'.format(interaction), end='')

	# end of game
	print('\ngame over')
	env.destroy()


if __name__ == "__main__":
	# create the env
	env = Maze()
	RL = DuelingDQN(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000)
	env.after(100, run_maze)
	env.mainloop()
	RL.plot_cost()
