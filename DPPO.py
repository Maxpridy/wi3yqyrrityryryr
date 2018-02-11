"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:4
tensorflow r1.3
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
import mujoco_py
from jaco_arm import JacoEnv

EP_MAX =  10000
EP_LEN = 256
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 10            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
#GAME = 'Pendulum-v0'
S_DIM, A_DIM = 21, 9         # state and action dimension


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        l1 = tf.layers.dense(self.tfs, 256, tf.nn.relu)
        l1 = tf.reshape(l1, [1, -1, 256])

        # RNN
        self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

        self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
        self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
        self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0, self.initial_lstm_state1)

        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]), np.zeros([1, 256]))

        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                        l1,
                                                        initial_state = self.initial_lstm_state,
                                                        sequence_length = [1],
                                                        time_major = False)

        self.lstm_outputs = tf.reshape(lstm_outputs, [-1,256])

        # critic
        l2 = tf.layers.dense(self.lstm_outputs, 128, tf.nn.relu)
        l3 = tf.layers.dense(l2, 128, tf.nn.relu)
        self.v = tf.layers.dense(l3, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())


    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op, { self.initial_lstm_state0: self.lstm_state_out[0],
                                                      self.initial_lstm_state1: self.lstm_state_out[1]})     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, { self.tfs: s, self.tfdc_r: r, self.initial_lstm_state0: self.lstm_state_out[0], self.initial_lstm_state1: self.lstm_state_out[1]})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv, self.initial_lstm_state0: self.lstm_state_out[0], self.initial_lstm_state1: self.lstm_state_out[1]}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r, self.initial_lstm_state0: self.lstm_state_out[0], self.initial_lstm_state1: self.lstm_state_out[1]}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 64, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 64, tf.nn.relu, trainable=trainable)
            l3 = tf.layers.dense(l2, 64, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l3, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l3, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s, self.initial_lstm_state0: self.lstm_state_out[0], self.initial_lstm_state1: self.lstm_state_out[1]})[0]
        return a
        #return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s, self.initial_lstm_state0: self.lstm_state_out[0], self.initial_lstm_state1: self.lstm_state_out[1]})[0, 0]


class Worker(object):
    def __init__(self, wid):

        self.wid = wid
        #self.env = gym.make(GAME).unwrapped
        self.env = JacoEnv(64, 64, 100)
        self.ppo = GLOBAL_PPO
        if self.wid == 0:
            self.viewer = mujoco_py.MjViewer(self.env.sim)

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data

                if self.wid == 0:
                    self.viewer.render()

                a = self.ppo.choose_action(s)
                s_, r, done = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)                    # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue

                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break
                    if done:
                        break

            with open("reward.txt", "a") as f:
                f.write(str(ep_r) + '\n')
            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            # r_d = 200 / (sum(GLOBAL_RUNNING_R[:-10:-1])/10 + 250 + GLOBAL_EP)
            # print(r_d)
            #self.env.reduce_rewarding_distance(r_d)
            #if sum(GLOBAL_RUNNING_R[:-11:-1])/10 > 100:
            #    self.env.reset_target()
            # if GLOBAL_EP > 1495 and GLOBAL_EP % 300 == 0:
            #     self.env.reset_target()
            # if GLOBAL_EP > 1495 and GLOBAL_EP % 300 == 1:
            #     self.env.reset_target()
            # if GLOBAL_EP > 1495 and GLOBAL_EP % 300 == 2:
            #     self.env.reset_target()
            # if GLOBAL_EP > 1495 and GLOBAL_EP % 300 == 3:
            #     self.env.reset_target()
            # if sum(GLOBAL_RUNNING_R[:-11:-1])/10 > 1500:
            #     with open("state.txt", "a") as f:
            #         f.write(str(self.env.sim.model.body_pos[-1]) + '\n')
            #         f.write(str(self.env.sim.model.geom_pos[-1]) + '\n')
            #print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)
            print(GLOBAL_EP, '/', EP_MAX, '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)

if __name__ == '__main__':

    phase = 1
    GLOBAL_PPO = PPO()

    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []  #Global_reward
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)



    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()

    env = JacoEnv(64, 64, 100)

    viewer = mujoco_py.MjViewer(env.sim)

    while True:
        s = env.reset()
        for t in range(200):
            viewer.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]
