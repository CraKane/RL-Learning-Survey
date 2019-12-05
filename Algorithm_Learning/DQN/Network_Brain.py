"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DQN:

	def __init__(self,
				 n_actions,
				 n_features,
				 learning_rate=0.01,
				 e_greedy=0.9,
				 reward_decay=0.9,
				 replace_target_iter=300,
				 memory_size=500,
				 batch_size=32,
				 e_greedy_increment=None,
				 output_graph=False):
		self.cost_his = []
		self.reward_his = []
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.e_greedy = e_greedy
		self.gamma = reward_decay
		self.memory_size = memory_size
		self.replace_tar_iter = replace_target_iter
		self.batch_size = batch_size
		self.e_greedy_increment = e_greedy_increment
		self.e_greedy = 0 if e_greedy_increment is not None else self.e_greedy
		self.replace_iter_step = 0

		# the total training step: episode
		self.total_training_step = 0

		# initialize zero memory [s, a, r, s_]
		self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

		# consist of [target_net, evaluate_net]
		self.build_net()
		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		# set a node operation: replace the t with e
		# assign: assign the first param with the second param
		self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		self.sess = tf.Session()

		if output_graph:
			# $ tensorboard --logdir=logs
			# tf.train.SummaryWriter soon be deprecated, use following
			tf.summary.FileWriter("logs/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())


	# build the 2 network
	def build_net(self):

		# ------------------ build evaluate_net ------------------
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
		self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
		# create 3 nodes
		with tf.variable_scope('eval_net'):
			# c_names(collections_names) are the collections to store variables
			c_names, n_l1, w_initializer, b_initializer = \
				['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
				tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

			# first layer. collections is used later when assign to target net
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

			# second layer. collections is used later when assign to target net
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_eval = tf.matmul(l1, w2) + b2

		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

		# ------------------ build target_net ------------------
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
		with tf.variable_scope('target_net'):
			# c_names(collections_names) are the collections to store variables
			c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

			# first layer. collections is used later when assign to target net
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

			# second layer. collections is used later when assign to target net
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_next = tf.matmul(l1, w2) + b2

	# save the (state, action)pair when it is playing
	def store_memory(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0

		transition = np.hstack((s, [a, r], s_))

		# replace the old memory with new memory
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition

		self.memory_counter += 1

	# learning & copy the params
	def update_copy_params(self):
		# check to replace target parameters
		if self.total_training_step % self.replace_tar_iter == 0:
			self.replace_iter_step = self.total_training_step / self.replace_tar_iter
			self.sess.run(self.replace_target_op)
			# print('target_params_replaced_{}'.format(self.replace_iter_step))

		# sample batch memory from all memory
		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		# q_next就是q_target，因此q_next的网络就是target网络
		# 从两个网络中获取batch采样的(state, action)pair的q值集合
		q_next, q_eval = self.sess.run(
			[self.q_next, self.q_eval],
			feed_dict={
				self.s_: batch_memory[:, -self.n_features:],  # 6size，其中最后两个是表达s_
				self.s: batch_memory[:, :self.n_features],  # 6size，其中前面两个是表达s
			})

		# change q_target w.r.t q_eval's action
		q_target = q_eval.copy()

		batch_index = np.arange(self.batch_size, dtype=np.int32)
		# 6size，其中索引为2的值，即memory中第三个值为动作值
		eval_act_index = batch_memory[:, self.n_features].astype(int)
		# 6size，其中索引为3的值，即memory中第四个值为环境奖励值
		reward = batch_memory[:, self.n_features + 1]
		# 选择q_next中动作估计值最大的动作，并使用它的估计值
		q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

		"""
		For example in this batch I have 2 samples and 3 actions:
		q_eval =
		[[1, 2, 3],
		 [4, 5, 6]]
		q_target = q_eval =
		[[1, 2, 3],
		 [4, 5, 6]]
		Then change q_target with the real q_target value w.r.t the q_eval's action.
		For example in:
			sample 0, I took action 0, and the max q_target value is -1;
			sample 1, I took action 2, and the max q_target value is -2:
		q_target =
		[[-1, 2, 3],
		 [4, 5, -2]]
		So the (q_target - q_eval) becomes:
		[[(-1)-(1), 0, 0],
		 [0, 0, (-2)-(6)]]
		We then backpropagate this error w.r.t the corresponding action to network,
		leave other action as error=0 cause we didn't choose it.
		"""

		# train eval network
		_, self.cost = self.sess.run([self._train_op, self.loss],
									 feed_dict={self.s: batch_memory[:, :self.n_features],
												self.q_target: q_target})

		# add the loss
		self.cost_his.append(self.cost)

		#
		self.e_greedy += 0.00001

		# increment the training step
		self.total_training_step += 1


	# choose the action according to the observation: s
	def choose_action(self, observation):
		# 在水平维扩充一维，即构成shape(1, x)，为了和占位符placeholder格式一致
		observation = observation[np.newaxis, :]

		if np.random.uniform() < self.e_greedy:
			# 我需要知道该状态下所有的动作估计值，而这个是eval网络得到的
			# 因此我用数据流将当前的observation喂进去得到动作估计值集合
			action_value = self.sess.run(self.q_next, feed_dict={self.s_ : observation})
			action = np.argmax(action_value)
		else:
			action = np.random.randint(0, self.n_actions)

		return action

	# plot the cost figure
	def plot_cost(self):
		plt.plot(np.arange(len(self.reward_his)), self.reward_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()

