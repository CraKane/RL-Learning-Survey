import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd

# 离散型倒立摆
GAME = 'CartPole-v0'
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()  #计算卡的cpu核个数
MAX_GLOBAL_EP = 500   #最大训练步数
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10  # 更新global net的迭代数 10次更新一回
GAMMA = 0.9  # 回报折扣率
ENTROPY_BETA = 0.001  # 熵的参数
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0  # 训练步数

env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                #导入状态值（观测值）
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                # 分别获取actor和critic的参数
                # 同时创建global_net
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                #导入所有状态值（观察值）
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                #。。。。。。
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                #导入v_target
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                #计算TD_ERROR
                td = tf.subtract(self.v_target, self.v, name='TD_error')

                #计算critic loss
                with tf.name_scope('c_loss'):
                    #计算td二次方的均值
                    self.c_loss = tf.reduce_mean(tf.square(td)) # critic的loss是平方loss
                #计算actor loss
                with tf.name_scope('a_loss'):
                    # Q * log（
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) *
                                             tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td) # 这里的td不再求导，当作是常数
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                #gradient descending
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            #同步动作
            with tf.name_scope('sync'):
                # 把主网络的参数赋予各子网络
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                # 使用子网络的梯度对主网络参数进行更新
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        # 初始化权值参数（0~1）
        w_init = tf.random_normal_initializer(0., .1)

        # 在global net中创建两个网络结构actor critic
        with tf.variable_scope('actor'):
            # 隐藏层使用Relu
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            # 得到每个动作的选择概率
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            # 隐藏层使用Relu
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            # 得到每个状态的价值函数value
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')

        #将actor和critic的所有权值参数收集起来

        # a参数和c参数由actor模块和critic模块而来
        # 自主地从两个模块中收集参数
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})  #newaxis == None
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action with regard to the actions prob
        return action


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        #create local_net
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        #只要没达到停止的条件，就持续学习下去
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                if self.name == 'W_0':
                    self.env.render()
                #挑选动作
                a = self.AC.choose_action(s)
                #得到下一个状态 回报等等
                s_, r, done, info = self.env.step(a)
                if done: r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)


                # update global and assign to local net
                if total_step % UPDATE_GLOBAL_ITER == 0 or done: #每10次更新
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    # reverse buffer r
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_ # 使用v(s) = r + v(s+1)计算target_v
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    #将子网络的参数给主网络的参数全部更新
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    #将主网络的参数全部给子网络进行下一波自主更新
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                #转化为下一个状态
                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(self.name, "Ep:", GLOBAL_EP, "| Ep_r: %i" % GLOBAL_RUNNING_R[-1])
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    # 设置接口句柄
    SESS = tf.Session()

    # 在cpu中创建一个global net
    with tf.device("/cpu:0"):
        # create the global_net
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')  # 使用RMSProp优化
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            #根据cpu核数创建worker - local_net
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
        # job(): worker.work()
        job = lambda: worker.work()  # 每一个线程让一个worker学习
        t = threading.Thread(target=job)  # 创建一个线程，并分配其工作
        t.start()  # 开启线程
        worker_threads.append(t)
    COORD.join(worker_threads)  # 把开启的线程加入主线程，(合并线程)等待threads结束

    # 将学习的结果组织成能够看懂的格式
    # res = np.concatenate([np.arange(len(GLOBAL_RUNNING_R)).reshape(-1,1),np.array(GLOBAL_RUNNING_R).reshape(-1,1)],axis=1)
    # pd.DataFrame(res, columns=['episode', 'a3c_reward']).to_csv('../a3c_reward.csv')
    # print(res)

    #描绘出学习越来越好的图像
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()