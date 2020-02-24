"""
@project = 20191119
@file = test_env
@author = 10374
@create_time = 2019/12/07 3:20
"""

import os
import gym
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env_ids = [spec.id for spec in gym.envs.registry.all()]
for env_id in env_ids:
	print(env_id)
	env = gym.make(env_id)
	for i_episode in range(20):
		observation = env.reset()
		for i in range(10):
			t = 0
			env.render()
			print(observation)
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
	env.close()