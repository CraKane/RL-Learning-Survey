"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

GAME = 'Pendulum-v0'
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()  # 计算cpu的核个数
MAX_EP_STEP = 200  # 一回合中最大的步数限制
MAX_GLOBAL_EP = 2000  # 最大训练步数
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10  # 更新global net的迭代数 10次更新一回
GAMMA = 0.9  # 回报折扣率
ENTROPY_BETA = 0.01  # 熵率
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0  # 训练步数

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            # 导入状态值（观测值）
            with tf.variable_scope(scope):
                # 分别获取actor和critic的参数
                # 同时创建global_net
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            # tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable的变量
            # tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                # td-error就是多步后计算的价值估计值减去当前网络下的同一个状态动作对的价值估计值
                td = tf.subtract(self.v_target, self.v, name='TD_error')

                # critic是value-based，所以critic的loss使用DQN的方法
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                # mu sigma是从网络中得到的，用正态分布又有exploration的作用
                # 调整mu sigma的偏置
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                # 计算actor的loss，它是policy-based
                with tf.name_scope('a_loss'):
                    # 将a_his扔进去，用频率法计算其对数概率
                    log_prob = normal_dist.log_prob(self.a_his)

                    # 将td截断，stop_grdient得到的当前被截断时得到的梯度
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                # use local params to choose action：模块的功能
                with tf.name_scope('choose_a'):
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])

                # 计算local的梯度，BP：back propagation
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                # 将global net的参数更新到local net里面
                with tf.name_scope('pull'):
                    # 将l_p更新为g_p-> global net参数代替local net的参数
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]

                # 将local网络的梯度传递给global net里面
                with tf.name_scope('push'):
                    # 直接使用RMSProp优化方法将得到的梯度进行更新
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        # 初始化权值参数（0~1）
        w_init = tf.random_normal_initializer(0., .1)

        # 在global net中创建两个网络结构actor critic
        with tf.variable_scope('actor'):
            # 隐藏层使用Relu
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            # ？？？
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            # 得到每个动作的选择概率
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            # 得到每个状态的价值函数value
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

        # a参数和c参数由actor模块和critic模块而来
        # 自主地从两个模块中收集参数
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    # A3C更新算法
    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        # 只要线程不中断，或者小于训练回合数，持续play
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            # reset the env
            s = self.env.reset()
            ep_r = 0  # 回合总奖励
            # 在单回合最大步数下进行训练
            for ep_t in range(MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)

                # 设立终止状态
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)    # normalize

                # 每10步更新一次，同时回合结束时也更新
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1

                # 一回合结束进行记录
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    # 设置接口句柄
    SESS = tf.Session()

    # 在cpu中创建一个global net
    with tf.device("/cpu:0"):
        # create the global_net
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')   # 使用RMSProp优化
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            # 根据cpu核数创建worker - local_net
            workers.append(Worker(i_name, GLOBAL_AC))

    # Coordinator类用来管理在Session中的多个线程，
    # 使用 tf.train.Coordinator()来创建一个线程管理器（协调器）对象。
    COORD = tf.train.Coordinator()  #训练中的一个对于线程的协调器，管理器
    # 初始化所有参数
    SESS.run(tf.global_variables_initializer())

    # 将建立的神经网络结构图保存
    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()  # 每一个线程让一个worker学习
        t = threading.Thread(target=job)  # 创建一个线程，并分配其工作
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)  # 把开启的线程加入主线程，(合并线程)等待threads结束

    # 将学习的结果组织成能够看懂的格式
    # res = np.concatenate([np.arange(len(GLOBAL_RUNNING_R)).reshape(-1,1),np.array(GLOBAL_RUNNING_R).reshape(-1,1)],axis=1)
    # pd.DataFrame(res, columns=['episode', 'a3c_reward']).to_csv('../a3c_reward.csv')
    # print(res)

    # 描绘出学习越来越好的图像
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

