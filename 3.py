import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy.matlib
from scipy.integrate import quad
from scipy import integrate
import multiprocessing as mp
import math
import numpy.matlib
import time
import os
os.environ['KMP_WARNINGS'] = '0'
UPDATE_GLOBAL_ITER = 20
ep_max =40
max_step=300
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001
LR_C = 0.001
N_S = 7
N_A = 9
UE_num = 6
MEMORY_SIZE = 3000
ACTION_SPACE = 9
Band = 0.72
packetsize = 0.000256
P_total = 10 ** 4.6 / 1000
band_total = 24
P_each_band = P_total / band_total
uplison=0.5
action_ori = [0,0,0,0,0,0]
delay_violation = [10 ** -5, 10 ** -4, 10 ** -5, 10 ** -3, 10 ** -3, 10 ** -4]
D_max = [1, 2, 3, 5, 2, 4]
SINR_ave =[10,10,10,10,10,10]
anna=2
SINR_e_ave = 3
SINR_ave_lin = np.zeros(UE_num)
class_ori = np.zeros(UE_num)
for i in range(UE_num):
    SINR_ave_lin[i] = round(10 ** (SINR_ave[i] / 10),2)
    if D_max[i]>1:
        class_ori[i]=2
    else:
        class_ori[i] = 1
SINR_e_ave_lin = round(10 ** (SINR_e_ave / 10),2)
Q_c = 4.7534
Q_e= 2.3263
eplison_c = 2 * 10 ** -6
action_list = np.arange(0,8,1)
n_PD_list = [1, 1, 1, 1, 1, 1, 1, 1, 1]
RB_num_list = [1, 2, 1, 2, 3, 1, 2, 3, 4]
W_real_list =[4, 8, 2, 4, 6, 1, 2, 3, 4]
W_total_list = [4, 8, 2, 4, 6, 2, 4, 6, 8]
time_list = [2, 2, 4, 4, 4, 4, 4, 4, 4]
TTI_list = [0.0000625, 0.0000625, 0.000125, 0.000125, 0.000125, 0.00025, 0.00025, 0.00025, 0.00025]
class env_snc():
    def __init__(self):
        self.agent = UE_num
        self.action = range(ACTION_SPACE)
        self.n_actions = ACTION_SPACE
        self.n_features = 7
        self.state = np.zeros(37)
        self.state_all = np.zeros(37)
        self.state_ = np.zeros(37)

        self.SINR_0 = np.ones(UE_num) * SINR_ave
        self.n_RE = np.zeros(UE_num)
        self.W_num = np.zeros(UE_num)
        self.SINR_lin = np.zeros(UE_num)
        for i in range(UE_num):
            self.SINR_lin[i] = 10 ** (self.SINR_0[i] / 10)
            if self.SINR_0[i] < -2.2:
                self.n_RE[i] = 288
            elif self.SINR_0[i] >= 4.2:
                self.n_RE[i] = 36
            elif self.SINR_0[i] < 0.2:
                self.n_RE[i] = 144
            else:
                self.n_RE[i] = 72
        self.h_rayleigh = np.random.rayleigh(1, UE_num)
        self._build_snc()

    def _build_snc(self):
        self.n_0 = 10 ** -17.4 / 1000
        self.P_total = 10 ** 4.6 / 1000
        self.BW = 2 * 10 ** 7
        self.T_c = np.zeros(UE_num)
        self.T_t = np.zeros(UE_num)
        self.calc = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.reward = 0

    def reset(self):
        W_sum = 0
        for i in range(self.agent):
            action_index = int(action_ori[i])
            W_num_real_total = W_total_list[action_index]
            self.state[i + 1] = class_ori[i]
            self.state[self.agent + 1 + i] = D_max[i]
            self.state[self.agent * 2 + 1 + i] = delay_violation[i]
            self.state[self.agent * 3 + 1 + i] = W_total_list[action_index]
            self.state[self.agent * 4 + 1 + i] = action_index
            self.state[self.agent * 5 + 1 + i] = SINR_ave[i]
            W_sum = W_sum + W_num_real_total
        self.state[0] = band_total - W_sum
        return self.state, action_ori

    def step(self, action_index, UE_index, observation):
        eplison_q = 1
        eplison_0 = 0
        reward = 1
        r = 0
        W_num_total = W_total_list[action_index]
        W_num_real = W_real_list[action_index]
        n_PD = n_PD_list[action_index]
        W_num_ori = observation[self.agent * 3 + 1 + UE_index]
        TTI_size = TTI_list[action_index]
        if class_ori[UE_index] == 1 and action_index == 32:
            reward = -2
            W_num_total = W_num_ori
            self.state_ = observation
            action = observation[self.agent * 4 + 1 + UE_index]
        elif observation[0] + W_num_ori < W_num_total:
            reward = 0
            W_num_total = W_num_ori
            self.state_ = observation
            action = observation[self.agent * 4 + 1 + UE_index]
        elif action_index == 32:
            reward = 0.5
            self.state_ = observation
            self.state_[0] = self.state[0] - W_num_ori
            self.state_[self.agent * 3 + 1 + UE_index] = 0
            action = action_index
        else:
            T_c = round(self.n_RE[UE_index] / 168 / RB_num_list[action_index] * TTI_size, 4)
            if T_c > TTI_size:
                T_c = TTI_size
            T_t = TTI_size - T_c
            if W_num_real == 0 or T_t == 0:
                r = 0
                eplison_q = 1
            else:
                N_k = Band * W_num_real * T_t * 1000
                aa = quad(lambda t: t ** (anna - 1) * math.e ** (-t), 0, np.inf)[0]
                aaa = SINR_e_ave_lin ** anna / aa
                V_e=round(math.sqrt((1 - (1 + SINR_e_ave_lin) ** -2)),2)
                r = round(Band*100000 * W_num_real * math.log((1 + SINR_ave_lin[UE_index]) / (1 + SINR_e_ave_lin) * math.e ** ((Q_c + Q_e * V_e )/ math.sqrt(N_k)/1000), 2),1)
                if TTI_size == 0.001:
                    D_max_q = round((D_max[UE_index] / 1000 - packetsize / r - TTI_size) * 1000, 2)
                else:
                    D_max_q = round(
                        (D_max[UE_index] / 1000 - time_list[action_index] * packetsize / r - TTI_size) * 1000, 2)
                if D_max_q < 0:
                    D_max_q = 0
                calc_before = [T_t * 1000, W_num_real, D_max_q, packetsize, SINR_ave[UE_index], SINR_e_ave, anna, Q_e]
                data = sio.loadmat('C:/Users/ZJX/Desktop/simulation/calc_record.mat')
                data1 = data['calc_record']
                data2 = numpy.array(data1)
                a = len(data2)
                for m in range(a):
                    b = [data2[m, 0:8]]
                    B = [[val in calc_before for val in n] for n in b]
                    if B == [[True, True, True, True, True, True, True, True]]:
                        eplison_q = data2[m, 8]
                        break
                if eplison_q == 1:
                    s_num = 100
                    theta = np.zeros(s_num)
                    H = np.zeros(s_num)
                    jifen = np.zeros((s_num, 2))
                    fenzi = np.zeros(s_num)
                    fenmu = np.zeros(s_num)
                    Y = np.zeros(s_num)
                    eplison_q_1 = np.zeros(s_num)
                    for j in range(s_num):
                        theta[j] = j * 0.005
                        H[j] = Band * W_num_real * theta[j] / math.log(2)

                        def f(gama_k, gama_e):
                            return ((1 + gama_e) * math.e ** (math.sqrt((1 - (1 + gama_e) ** (-2)) / N_k) * Q_c)) ** H[
                                j] * aaa * gama_e ** (anna - 1) * math.e ** (-SINR_e_ave_lin * gama_e) * gama_k ** (
                                       -H[j]) * math.e ** (-gama_k)
                        def h(gama_e):
                            return (gama_e + 1) / SINR_ave_lin[UE_index]
                        jifen[j] = integrate.dblquad(f, 0, np.inf, 0, np.inf)
                        Y[j] = SINR_ave_lin[UE_index] ** (-H[j]) * math.e ** (
                                    math.sqrt(1 / N_k) * Q_e * H[j] + 1 / SINR_ave_lin[UE_index]) * jifen[j, 0]
                        fenzi[j] = (1 - eplison_c) * Y[j] + eplison_c
                        fenmu[j] = 1 - math.e ** (packetsize * (math.e ** theta[j] - 1)) * fenzi[j]
                        eplison_q_1[j] = fenzi[j] ** D_max_q / fenmu[j]
                    for k in range(s_num - 1):
                        if eplison_q_1[k] < 0 and eplison_q_1[k + 1] > 0:
                            theta_50 = theta[k]
                            break
                    if k != 98:
                        for j in range(s_num):
                            theta[j] = j * 0.00005 + theta_50
                            H[j] = Band * W_num_real * theta[j] / math.log(2)

                            def f(gama_k, gama_e):
                                return ((1 + gama_e) * math.e ** (math.sqrt((1 - (1 + gama_e) ** (-2)) / N_k) * Q_c)) ** \
                                       H[j] * aaa * gama_e ** (anna - 1) * math.e ** (
                                                   -SINR_e_ave_lin * gama_e) * gama_k ** (-H[j]) * math.e ** (-gama_k)
                            def h(gama_e):
                                return (gama_e + 1) / SINR_ave_lin[UE_index]
                            jifen[j] = integrate.dblquad(f, 0, np.inf, 0, np.inf)
                            Y[j] = SINR_ave_lin[UE_index] ** (-H[j]) * math.e ** (
                                        math.sqrt(1 / N_k) * Q_e * H[j] + 1 / SINR_ave_lin[UE_index]) * jifen[j, 0]
                            fenzi[j] = (1 - eplison_c) * Y[j] + eplison_c
                            fenmu[j] = 1 - math.e ** (packetsize * (math.e ** theta[j] - 1)) * fenzi[j]
                            eplison_q_1[j] = fenzi[j] ** D_max_q / fenmu[j]
                    eplison_q_1[eplison_q_1 <= 0] = 10
                    eplison_q = min(eplison_q_1)
                    calc = [T_t * 1000, W_num_real, D_max_q, packetsize, SINR_ave[UE_index], SINR_e_ave, anna, Q_e,
                            eplison_q]
                    data2 = np.row_stack((data2, calc))
                    sio.savemat('C:/Users/ZJX/Desktop/simulation/calc.mat', {'calc': data2})
                if eplison_q > 1 or eplison_q < 0:
                    eplison_q = 1
            eplison_0 = (eplison_q + eplison_c) ** (n_PD + 1)
            if eplison_0 > delay_violation[UE_index]:
                reward = -1
            self.state_[0] = observation[0] - W_num_total + W_num_ori
            for i in range(self.agent):
                self.state_[1 + i] = observation[1 + i]
                self.state_[self.agent + 1 + i] = observation[self.agent + 1 + i]
                self.state_[self.agent * 2 + 1 + i] = observation[self.agent * 2 + 1 + i]
                self.state_[self.agent * 4 + 1 + i] = observation[self.agent * 4 + 1 + i]
                self.state_[self.agent * 5 + 1 + i] = observation[self.agent * 5 + 1 + i]
                if i == UE_index:
                    self.state_[self.agent * 3 + 1 + i] =  W_num_total
                else:
                    self.state_[self.agent * 3 + 1 + i] = observation[self.agent * 3 + 1 + i]
            action = action_index
        self.reward = reward
        return self.state_, self.reward, eplison_0, r, n_PD, W_num_total, action, self.calc, TTI_size


class ACNet(object):
    sess = None
    def __init__(self, scope, opt_a=None, opt_c=None, global_net=None):
        if scope == 'global_net':  # get global network
            with tf.compat.v1.variable_scope(scope):
                self.s = tf.compat.v1.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            with tf.compat.v1.variable_scope(scope):
                self.s = tf.compat.v1.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.compat.v1.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.compat.v1.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(
                        tf.math.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                        axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.math.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
            self.global_step = tf.compat.v1.train.get_or_create_global_step()
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = opt_a.apply_gradients(zip(self.a_grads, global_net.a_params), global_step=self.global_step)
                    self.update_c_op = opt_c.apply_gradients(zip(self.c_grads, global_net.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.compat.v1.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.compat.v1.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def choose_action(self, s):  # run by a local
        s=np.array(s)
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def update_global(self, feed_dict):
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

def work(job_name, task_index, global_ep, lock, r_queue, global_running_r):

    EE_norm_record=[]
    SE_norm_record=[]
    EE_norm_max = 0
    SE_norm_max = 0
    # set work's ip:port
    cluster = tf.train.ClusterSpec({
        "ps": ['localhost:2220', 'localhost:2221',],
        "worker": ['localhost:2222', 'localhost:2223', 'localhost:2224', 'localhost:2225',]
    })
    server = tf.distribute.Server(cluster, job_name=job_name, task_index=task_index)
    if job_name == 'ps':
        print('Start Parameter Sever: ', task_index)
        server.join()
    else:
        t1 = time.time()
        env = env_snc()
        print('Start Worker: ', task_index)
        with tf.device(tf.compat.v1.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
            opt_a = tf.compat.v1.train.RMSPropOptimizer(LR_A, name='opt_a')
            opt_c = tf.compat.v1.train.RMSPropOptimizer(LR_C, name='opt_c')
            global_net = ACNet('global_net')
        local_net = ACNet('local_ac%d' % task_index, opt_a, opt_c, global_net)
        # set training steps
        hooks = [tf.estimator.StopAtStepHook(last_step=100000)]
        with tf.compat.v1.train.MonitoredTrainingSession(master=server.target, is_chief=True, hooks=hooks,) as sess:
            print('Start Worker Session: ', task_index)
            local_net.sess = sess
            total_step = 1
            buffer_s, buffer_a, buffer_r = [], [], []
            step = 0
            reward_set = np.zeros(UE_num)
            TTI_size_set = np.zeros(UE_num)
            r_set = np.zeros(UE_num)
            n_PD_set = np.zeros(UE_num)
            eplison_set = np.zeros(UE_num)
            W_num_total_set = np.zeros(UE_num)
            eplison_c_set = np.zeros(UE_num)
            reward_set_step = np.zeros(UE_num)
            TTI_size_set_step = np.zeros(UE_num)
            r_set_step = np.zeros(UE_num)
            n_PD_set_step = np.zeros(UE_num)
            eplison_set_step = np.zeros(UE_num)
            W_num_total_set_step = np.zeros(UE_num)
            action_set_step = np.zeros(UE_num)
            eplison_c_set_step = np.zeros(UE_num)
            step_set = []
            reward_sum_set = []
            reward_ave_set = []

            while global_ep.value < ep_max:
                observation, action_ori = env.reset()
                ep_r = 0
                for step1 in range(max_step):
                    step = step + 1
                    if step1 == 0:
                        action_index_set = action_ori
                        action_index_set_0 = action_ori
                    reward_step = []
                    TTI_size_step = []
                    r_step = []
                    n_PD_step = []
                    W_num_total_step = []
                    eplison_step = []
                    eplison_c_step = []
                    observation_0_set = np.zeros(37)
                    observation_set = np.zeros(37)
                    for i in range(UE_num):
                        UE_index = i
                        action_index = local_net.choose_action( [observation[0], observation[i + 1], observation[UE_num + i + 1],observation[UE_num * 2 + i + 1],observation[UE_num * 3 + i + 1],observation[UE_num * 4 + i + 1],observation[UE_num *5 + i + 1]])
                        action_index_set_0[i] = action_index
                        observation_0_set = np.row_stack((observation_0_set, observation))
                        observation_, reward, eplison_0, r, n_PD, W_num_total, action, calc, TTI_size = env.step(action_index, UE_index, observation)
                        observation_set = np.row_stack((observation_set, observation_))
                        action_index_set[i] = action
                        reward_step.append(reward)
                        TTI_size_step.append(TTI_size*1000)
                        r_step.append(r)
                        n_PD_step.append(n_PD)
                        eplison_step.append(eplison_0)
                        W_num_total_step.append(W_num_total)
                        eplison_c_step.append(eplison_c * 10 ** 8)
                        observation = observation_
                        if i == UE_num - 1:
                            for j in range(UE_num):
                                observation_[j + UE_num + 1] = action_index_set[j]
                    reward_sum = sum(reward_step)
                    W_num_sum = sum(W_num_total_step)
                    reward_set_step = np.row_stack((reward_set_step, reward_step))
                    TTI_size_set_step = np.row_stack((TTI_size_set_step, TTI_size_step))
                    r_set_step = np.row_stack((r_set_step, r_step))
                    n_PD_set_step = np.row_stack((n_PD_set_step, n_PD_step))
                    eplison_set_step = np.row_stack((eplison_set_step, eplison_step))
                    W_num_total_set_step = np.row_stack((W_num_total_set_step, W_num_total_step))
                    action_set_step = np.row_stack((action_set_step, action_index_set))
                    eplison_c_set_step = np.row_stack((eplison_c_set_step, eplison_c_step))
                    EE_fenmu = 0
                    EE_fenzi = 0
                    SE_fenmu = 0
                    SE_fenzi = 0
                    EE_norm = 0
                    SE_norm = 0
                    penalty_num = 0
                    penalty = 0
                    for j in range(UE_num):
                        action = int(action_index_set[j])
                        if action == 32:
                            penalty = 0.1
                        if reward_step[j] == 1 or reward_step[j] == 0.5:
                            EE_fenzi = EE_fenzi + 256
                            SE_fenzi = SE_fenzi + r_step[j]
                        EE_fenmu = EE_fenmu + W_total_list[action] * P_each_band * time_list[action] * 0.0000625
                        SE_fenmu = SE_fenmu + W_total_list[action] * Band * 1000000
                        penalty_num = penalty_num + penalty
                    if EE_fenzi == 0 or EE_fenmu == 0:
                        EE_norm = 0
                    elif SE_fenzi == 0 or SE_fenmu == 0:
                        SE_norm = 0
                    else:
                        EE_norm_record.append(EE_fenzi / EE_fenmu)
                        SE_norm_record.append(SE_fenzi / SE_fenmu)
                        EE_norm_max = max(EE_norm_record)
                        SE_norm_max = max(SE_norm_record)
                        if EE_fenzi / EE_fenmu > 530000 or SE_fenzi / SE_fenmu > 0.6:
                            bbbbbbbbbbbbbbbbbbbb = 1
                        EE_norm = round(EE_fenzi / EE_fenmu / 530000, 2)
                        SE_norm = round(SE_fenzi / SE_fenmu / 0.6, 2)
                        if EE_norm > 1 or EE_norm > 1:
                            EE_norm = 1
                            SE_norm = 1
                    reward_sum = round((EE_norm ** uplison) * (SE_norm ** (1 - uplison)) - penalty_num,3)
                    for i in range(UE_num):
                        buffer_s.append([observation_0_set[i+1,0],observation_0_set[i+1,i+1],observation_0_set[i+1,UE_num+i+1],observation_0_set[i+1,UE_num*2+i+1],observation_0_set[i+1,UE_num*3+i+1],observation_0_set[i+1,UE_num*4+i+1],observation_0_set[i+1,UE_num*5+i+1]])  # 将当前状态，行动和回报加入缓存
                        buffer_a.append(action_index_set_0[i])
                        buffer_r.append(reward_sum*10)
                    done = False
                    if step1 == max_step-1 :
                        done = True
                    ep_r += reward_sum
                    reward_ave_1=round(ep_r / max_step,3)
                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                        if done:
                            v_s_ = 0
                        else:
                            observation_ = np.array([observation_[0],observation_[i+1],observation_[UE_num+i+1],observation_[UE_num*2+i+1],observation_[UE_num*3+i+1],observation_[UE_num*4+i+1],observation_[UE_num*5+i+1]])
                            v_s_ = sess.run(local_net.v, {local_net.s: observation_[np.newaxis, :]})[0, 0]
                        buffer_v_target = []
                        for r in buffer_r[::-1]:  # reverse buffer r
                            v_s_ = r + GAMMA * v_s_
                            buffer_v_target.append(v_s_)
                        buffer_v_target.reverse()
                        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                            buffer_v_target)
                        feed_dict = {
                            local_net.s: buffer_s,
                            local_net.a_his: buffer_a,
                            local_net.v_target: buffer_v_target,
                        }
                        local_net.update_global(feed_dict)
                        buffer_s, buffer_a, buffer_r = [], [], []
                        local_net.pull_global()
                    total_step += 1
                    if done:
                        if global_ep.value > ep_max - 1:
                            time.sleep(10000)
                        r_queue.put(reward_ave_1)
                        print(
                            "Task: %i" % task_index,
                            "| Ep: %i" % global_ep.value,
                            "| Reward_ave: %.3f" % reward_ave_1,
                        )
                        with lock:
                            global_ep.value += 1
                        break
        t2 = time.time() - t1
        print('Worker Done: ', task_index, t2)
        re = []
        step = []
        reward_ave = []
        r_parameter_set_0=[]
        W_num_1 = np.zeros(UE_num)
        TTI_size_1 = np.zeros(UE_num)
        n_PD_1 = np.zeros(UE_num)
        eplison_1 = np.zeros(UE_num)
        while not r_queue.empty():
            reward_ave.append(r_queue.get())
        if len(reward_ave) > 1:
            plt.plot(np.arange(len(reward_ave)), reward_ave)
            plt.title('reward_ave')
            plt.xlabel('Episode')
            plt.ylabel('reward_ave')
            plt.show()
        sio.savemat('C:/Users/ZJX/Desktop/simulation/4_fixedPD_1_10_3.mat', {'reward_set': re, 'reward_ave': reward_ave, 'W_num': W_num_1, 'TTI_size': TTI_size_1, 'n_PD': n_PD_1, 'eplison': eplison_1,'time': t2,'reward_ave_set':reward_ave,
                                                                              'r_parameter_set':r_parameter_set_0,'eplison_c_set':eplison_c_set})

if __name__ == "__main__":
    global_ep = mp.Value('i', 0)
    lock = mp.Lock()
    r_queue = mp.Queue()
    global_running_r = mp.Value('d', 0)
    jobs = [
        ('ps', 0), ('ps', 1),
        ('worker', 0), ('worker', 1), ('worker', 2), ('worker', 3)
    ]
#
    t1 = time.time()
    ps = [mp.Process(target=work, args=(j, i, global_ep, lock, r_queue, global_running_r), ) for j, i in jobs]
    [p.start() for p in ps]
    [p.join() for p in ps[2:]]
    ep_r = []
    while not r_queue.empty():
        ep_r.append(r_queue.get())
    plt.plot(np.arange(len(ep_r)), ep_r)
    plt.title('Distributed training')
    plt.xlabel('Step')
    plt.ylabel('Total moving reward')
    plt.show()
