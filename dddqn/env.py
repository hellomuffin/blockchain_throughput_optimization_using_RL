import numpy as np

from torch import smm

# \\wsl$\Ubuntu\usr\local\lib\python3.6\dist-packages\gym\envs\classic_control

# https://blog.csdn.net/lxs3213196/article/details/110080667

# https://github.com/openai/gym

# @misc{1606.01540,
#   Author = {Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba},
#   Title = {OpenAI Gym},
#   Year = {2016},
#   Eprint = {arXiv:1606.01540},
# }

"""
Parameters
"""

Pack_coef = 1

Total_spectrum = 28
Total_PCG = 30
Total_FCG = 6
Total_node = Total_FCG + Total_PCG
BadNodeProb = 0.25

P_min = 0.6
P_max = 1.0
#
mu = 150
#
Min_GroupBlock = 6


def p_(n, x):
    val = 0
    for i in range(0, int((n - 1) / 3) + 1):
        val += (np.math.factorial(n)) / (np.math.factorial(i) * np.math.factorial(n - i)) * pow(x, i) * pow(1 - x,
                                                                                                            n - i)
    return val


def P(m, n, x):  # 安全的概率
    val = 0
    for i in range(0, int((n - 1) / 3) + 1):
        val += (np.math.factorial(n)) / (np.math.factorial(i) * np.math.factorial(n - i)) * pow(
            1 - p_(np.math.floor(m / n), x), i) * pow(p_(np.math.floor(m / n), x), n - i)
    return val


def MarkovStep(curr, ProbMatrix):
    r = np.random.uniform(0, 1)
    for i in range(len(ProbMatrix[curr])):
        if r > ProbMatrix[curr][i]:
            r -= ProbMatrix[curr][i]
        else:
            curr = i
            break
    return curr


class MecBCEnv1:
    def __init__(self, SM=None, DI=None):
        self.state_size = 2
        self.action_size = 3  # 动作空间维度：D(区块间隔)，Sm(miniblock大小)，K(分组数量)

        self.CompRes_size = 3  # Computation Resource
        self.CompRes = np.array([200, 500, 1000])
        self.CompRes_trans = np.array([[0.7, 0.2, 0.1], [0.2, 0.1, 0.7], [0.1, 0.7, 0.2]])

        self.Sinr_size = 5  # SINR
        self.Sinr = np.array([1, 3, 7, 15, 31])
        self.Sinr_trans = np.array([[0.6, 0.4, 0.2, 0.1, 0.05],
                                    [0.05, 0.6, 0.4, 0.2, 0.1],
                                    [0.1, 0.05, 0.6, 0.4, 0.2],
                                    [0.2, 0.1, 0.05, 0.6, 0.4],
                                    [0.4, 0.2, 0.1, 0.05, 0.6]])
        self.Sinr_trans = self.Sinr_trans / 1.35

        # self.action_lower_bound = [0.5, 4, 1]
        # self.action_higher_bound = [4, np.math.floor(Total_PCG/Min_GroupBlock), 4]
        self.observation_space = [self.CompRes, self.Sinr]  # PCG, FCG, user
        self.Kspace = []
        for k in range(1, np.math.floor(Total_PCG / Min_GroupBlock) + 1):
            pr = p_(int(Total_PCG / k), BadNodeProb)
            print(pr)
            if pr >= P_min:
                self.Kspace.append(k)
        print(self.Kspace)
        #
        if SM == None:
            self.SMspace = [0.5 * i for i in range(1, 9)]
        else:
            self.SMspace = [SM]
        if DI == None:
            self.DIspace = [0.5 * i for i in range(1, 9)]
        else:
            self.DIspace = [DI]
        #
        self.state = [[0, 0] for i in range(Total_node)]
        # self.normalized_state = [[0, 0] for i in range(Total_node + Total_user)]
        inputt = np.array(self.state).reshape([-1])
        self.input_state = list(inputt[:Total_node])
        self.input_state.extend(list(inputt[-Total_node:]))
        # print(self.input_state)
        self.reset()
        self.epoch = 0

    def reset(self):
        self.epoch = 0
        for i in range(Total_PCG):  # PCG + FCG
            self.state[i][0] = np.random.randint(0, self.CompRes_size)
            # self.normalized_state[i][0] = (self.state[i][0] - 200) / 800

        for i in range(Total_node):  # PCG + FCG + User
            self.state[i][1] = np.random.randint(0, self.Sinr_size)
            # self.normalized_state[i][1] = (self.state[i][1] - 1) / 30

        inputt = np.array(self.state).reshape([-1])
        self.input_state = list(inputt[:Total_PCG])
        # print(len(self.input_state))
        self.input_state.extend(list(inputt[-(Total_node):]))
        # print(len(self.input_state))

    def step(self, action):
        '''
        action格式（DQN)：三元素列表，表示各变量取值。取值转化在神经网络中完成。
        '''
        # for i in range(self.action_size):  #K并非连续
        #     action[i] = action[i] * (self.action_higher_bound[i] - self.action_lower_bound[i]) + self.action_lower_bound[i]

        # Calculate DataRate in __init__ will be more efficient.
        DataRate = [Total_spectrum / (Total_node) * np.math.log(self.Sinr[self.state[i][1]] + 1) / np.math.log(2) for i
                    in range(Total_node)]
        DataRate = np.array(DataRate)
        DataRate_PCGMin = np.min(DataRate[0: Total_PCG])
        # Not accurate if we do not identify the primary node

        CompRes_Min = np.min(np.array([self.CompRes[self.state[i][0]] for i in range(Total_PCG)]))
        t_PAK1 = action[0] / CompRes_Min
        t_PBFT1 = 5 * action[0] / DataRate_PCGMin
        t_DEL1 = action[0] / DataRate_PCGMin
        DataRate_FCGMin = np.min(DataRate[Total_PCG: Total_node])
        t_PBFT2 = action[0] * action[1] / DataRate_PCGMin + 4 * action[0] * action[1] / DataRate_FCGMin
        TTF = t_DEL1 + t_PAK1 + t_PBFT1 + t_PBFT2 + action[2]
        # print(TTF, action[2])
        C2 = bool(P(Total_node, action[1], BadNodeProb) >= P_min and P(Total_node, action[1], BadNodeProb) <= P_max)  #安全性
        C1 = bool(TTF <= mu * action[2])  #时延

        if (C1 and C2 and action[2] > 0):
            reward = action[0] * action[1] / action[2]
        else:
            reward = 0
            print(C1, C2)

        # print(reward)

        self.epoch += 1
        done = 0

        if self.epoch > 100:
            self.reset()
            done = 1

        # 状态转移
        for i in range(Total_PCG):
            self.state[i][0] = MarkovStep(self.state[i][0], self.CompRes_trans)
            # self.normalized_state[i][0] = (self.state[i][0] - 200) / 800

        for i in range(Total_node):
            self.state[i][1] = MarkovStep(self.state[i][1], self.Sinr_trans)
            # self.normalized_state[i][1] = (self.state[i][1] - 1) / 30

        inputt = np.array(self.state).reshape([-1])
        self.input_state = list(inputt[:Total_PCG])
        self.input_state.extend(list(inputt[-Total_node:]))

        return self.input_state, reward, done, {}


if __name__ == '__main__':
    a = MecBCEnv1()
    print(len(a.input_state))
    # for i in a.SMspace:
    #     for j in a.Kspace:
    #         for k in a.DIspace:
    #             a.step([i, j, k])
