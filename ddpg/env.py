import numpy as np

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
Total_user = 8
BadNodeProb = 0.25

P_min = 0.6
P_max = 1.0
mu = 150

Min_GroupBlock = 6


def p_(n, x):
    val = 0
    for i in range(0, int((n - 1) / 3) + 1):
        val += (np.math.factorial(n)) / (np.math.factorial(i) * np.math.factorial(n - i)) * pow(x, i) * pow(1 - x,
                                                                                                            n - i)
    return val


def P(m, n, x):
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


class MecBCEnv(object):
    def __init__(self):
        self.state_size = 2
        self.action_size = 3

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

        # self.observation_space = [spaces.Tuple((spaces.Discrete(self.CompRes_size),       #CompRes
        #                                         spaces.Discrete(self.Sinr_size)))         #Sinr
        #                                         for i in range(Total_node + Total_user)]  # PCG, FCG, user

        # self.action_space = spaces.Tuple((spaces.Box(low=0.5, high=4, shape=1),         #SM, small blocksize
        #                                     spaces.Box(low = 1.0, high = np.math.floor(Total_PCG/Min_GroupBlock), shape = 1),   #K, number of partition.
        #                                     spaces.Box(low=0.5, high=4, shape=1)))      #DI, block interval

        self.action_lower_bound = [0.5, 1, 1]
        self.action_higher_bound = [3.7, np.math.floor(Total_PCG / Min_GroupBlock), 4]

        self.available_K = []

        for i in range(1, int(Total_PCG / 6) + 1):
            if P(Total_PCG, i, BadNodeProb) > 0.6:
                self.available_K.append(i)

        # the actor assume the action space lower bound == 0

        self.state = np.array([0 for i in range(2 * Total_node + Total_user)])
        # first Total_node states are Computering Resource, and the remaining (Total_node + Total_user) state are SINR

        self.epoch = 0

    def reset(self):
        self.epoch = 0
        for i in range(Total_node):  # PCG + FCG
            self.state[i] = np.random.randint(0, self.CompRes_size)

        for i in range(Total_node + Total_user):  # PCG + FCG + User
            self.state[i + Total_node] = np.random.randint(0, self.Sinr_size)

        State_ = []
        for i in range(Total_node):
            State_.append(self.CompRes[self.state[i]] / 1000)
        for i in range(Total_node + Total_user):
            State_.append(self.Sinr[self.state[i + Total_node]] / 31)

        State_ = np.array(State_)

        return State_

    def step(self, action):
        for i in range(self.action_size):
            # action[i] = np.math.tanh(action[i])
            if action[i] < -1: action[i] = -1
            if action[i] > 1: action[i] = 1

        action[0] = (1 + action[0]) * (self.action_higher_bound[0] - self.action_lower_bound[0]) / 2 + \
                    self.action_lower_bound[0]
        action[2] = (1 + action[2]) * (self.action_higher_bound[2] - self.action_lower_bound[2]) / 2 + \
                    self.action_lower_bound[2]

        index = int((1 + action[1]) / 2 * len(self.available_K) - 0.000001)
        action[1] = self.available_K[index]

        # print(action[0], action[1], action[2])

        # Calculate DataRate in __init__ will be more efficient.
        DataRate = [Total_spectrum / (Total_node + Total_user) * np.math.log(
            self.Sinr[self.state[i + Total_node]] + 1) / np.math.log(2) for i in range(Total_user + Total_node)]

        DataRate = np.array(DataRate)

        DR1 = DataRate[Total_node: Total_node + Total_user]
        DataRate_UserMin = DR1.min()

        DR2 = DataRate[0: Total_PCG]
        DataRate_PCGMin = DR2.min()

        # Not accurate if we do not identify the primary node

        CR = np.array([self.CompRes[self.state[i]] for i in range(Total_user)])
        CompRes_Min = CR.min()

        t_PAK1 = action[0] / CompRes_Min

        t_PBFT1 = action[0] / DataRate_UserMin + 4 * action[0] / DataRate_PCGMin

        t_DEL1 = action[0] / DataRate_PCGMin

        DataRate_FCGMin = np.min(DataRate[Total_PCG: Total_node])

        t_PBFT2 = action[0] * action[1] / DataRate_PCGMin + 4 * action[0] * action[1] / DataRate_FCGMin

        TTF = t_DEL1 + t_PAK1 + t_PBFT1 + t_PBFT2 + action[2]

        C1 = bool(P(Total_PCG, action[1], BadNodeProb) >= P_min and P(Total_PCG, action[1], BadNodeProb) <= P_max)
        C2 = bool(TTF <= mu * action[2])
        # C2 = True

        if (C1 and C2 and action[2] > 0):
            reward =  action[0] * action[1] / action[2] - 5
        else:
            reward = -5
            print(C1, C2)

        self.epoch += 1
        done = 0

        for i in range(Total_node):
            self.state[i] = MarkovStep(self.state[i], self.CompRes_trans)

        for i in range(Total_node + Total_user):
            self.state[i + Total_node] = MarkovStep(self.state[i + Total_node], self.Sinr_trans)

        if self.epoch > 100:
            self.reset()
            done = 1

        State_ = []
        for i in range(Total_node):
            # State_.append(self.CompRes[self.state[i]]/1000)
            State_.append(self.state[i] / 3)
        for i in range(Total_node + Total_user):
            State_.append(self.state[i + Total_node] / 5)

        State_ = np.array(State_)

        # print(State_[0], State_[30], State_[60])
        return State_, reward, done, C2, action


