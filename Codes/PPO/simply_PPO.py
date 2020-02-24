"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000  # 最大回合数
EP_LEN = 200   # 回合最大步数
GAMMA = 0.9    # 折扣率
A_LR = 0.0001  # actor学习率
C_LR = 0.0002  # critic学习率
BATCH = 32     # batch大小
A_UPDATE_STEPS = 10  # actor一次更新10步
C_UPDATE_STEPS = 10  # critic一次更新10步
S_DIM, A_DIM = 3, 1  # 状态维度和动作维度
# 方法选择：1. 自适应KL系数  2. 裁切方法
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty，目标值为0.01，
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective，ε=0.2
                                                    # find this is better
][1]        # choose the method for optimization


class PPO(object):

    def __init__(self):
        # 设置连接句柄
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            # Relu，100个神经元的输入层
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            # critic loss是均方误差优势函数
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            # 使用Adam优化器最小化c loss
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        with tf.variable_scope('actor'):
            # actor构建两个策略网络，Π和Πold
            pi, pi_params = self._build_anet('pi_actor', trainable=True)
            oldpi, oldpi_params = self._build_anet('oldpi_actor', trainable=False)

            # 定义两个操作符，选动作操作和更新参数操作
            # choosing action
            with tf.variable_scope('sample_action'):
                self.choose_action_op = tf.squeeze(pi.sample(1), axis=0)

            # 使用最新的策略参数更新旧的策略参数
            with tf.variable_scope('update_oldpi'):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            # 定义动作占位符结点
            self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
            # 定义优势函数占位符结点：作用？？？
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
            # 定义a loss
            with tf.variable_scope('loss'):
                # 能使得策略越来越好的单调部分
                with tf.variable_scope('surrogate'):
                    # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                    ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                    surr = ratio * self.tfadv

                # 两种方法界定KL惩罚
                if METHOD['name'] == 'kl_pen':
                    self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                    kl = tf.distributions.kl_divergence(oldpi, pi)
                    self.kl_mean = tf.reduce_mean(kl)
                    # a loss是L函数的期望
                    # 希望最小化a loss
                    self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
                else:   # clipping method, find this is better
                    self.aloss = -tf.reduce_mean(tf.minimum(
                        surr,
                        tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

            # 使用Adam优化器最小化a loss
            with tf.variable_scope('atrain'):
                self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        # 保存网络结构图
        tf.summary.FileWriter("log/", self.sess.graph)

        # 初始化所有参数
        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        # 连接结点，进行更新参数操作
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        '''adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful'''

        # update actor
        if METHOD['name'] == 'kl_pen':
            # 批量更新，更新10次
            for _ in range(A_UPDATE_STEPS):
                _, self.kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if self.kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if self.kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            # 批量更新，更新10次
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]  # 批量更新，更新10次

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            # Relu 100个神经元的输入层
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        # 输出最后的分布结果和参数
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.choose_action_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    # 从critic中得到value
    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []

# 学习更新过程
for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # 更新ppo的过程
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            # 一旦一个批量batch已满，或者超过回合的最大步数，则更新参数(策略)
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)

    # 插入总奖励的特技，类似于取平均
    if ep == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %.4f" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )


# 展示学习进步过程
plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()
